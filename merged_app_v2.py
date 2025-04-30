import os, time, math, logging, base64, re, itertools
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO          # used only for fallback object-ness
import mediapipe as mp
from openai import OpenAI

# ------------------------------ CONSTANTS ----------------------------------
BOUNDARY_Y            = 550           # y-coord of the red boundary line
ZOOM_FACTOR           = 1.1           # digital zoom
CONF_THRESHOLD        = 0.30          # YOLO conf
MAX_LOST_FRAMES       = 5             # drop track after N missing frames
MAX_TRACK_DISTANCE    = 200           # px radius to associate detections
TRACK_HISTORY         = 10            # stored points for velocity calc
EMIT_INTERVAL         = 0.5           # sec between detection emit
SCREENSHOT_INTERVAL   = 1.0           # sec between screenshots
PERSON_CLS_ID         = 0             # COCO id for “person”
CROP_SCALE = 4          # 1.0 = just the hand, 2.0 = double width/height
CROP_MIN_PAD = 100         # absolute px pad if scale still ends up tiny

# ------------------------------ GLOBALS ------------------------------------
logging.basicConfig(level=logging.INFO)
logger  = logging.getLogger(__name__)

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet',
                    logger=True, engineio_logger=True)

# YOLO object-ness model (class-agnostic)
yolo_obj = YOLO("yolo11n.pt")         # auto-download on first run

# MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35
)

# OpenAI

# Inventory / tracking state
inventory, tracks = {}, {}
next_track_id     = itertools.count()      # generator: 0,1,2,…

# --------------------------- HELPER FUNCS ----------------------------------
def digital_zoom(frame, factor):
    h, w = frame.shape[:2]
    nw, nh = int(w/factor), int(h/factor)
    x1, y1 = (w-nw)//2, (h-nh)//2
    crop   = frame[y1:y1+nh, x1:x1+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

def hands_in_frame(bgr):
    """Return list of hand boxes [x1,y1,x2,y2]."""
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res   = mp_hands.process(rgb)
    boxes = []
    if res.multi_hand_landmarks:
        h, w = bgr.shape[:2]
        for hand in res.multi_hand_landmarks:
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            x1, y1, x2, y2 = min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h
            boxes.append((x1, y1, x2, y2))
    return boxes

def add_to_inventory(label, image=None):
    if label not in inventory:
        inventory[label] = {'count': 0, 'images': [], 'last_updated': time.time()}
    inventory[label]['count'] += 1
    if image is not None:
        inventory[label]['images'].append(image)
    inventory[label]['last_updated'] = time.time()
    logger.info(f"Added {label}. Count: {inventory[label]['count']}")

def remove_from_inventory(label):
    if label in inventory and inventory[label]['count'] > 0:
        inventory[label]['count'] -= 1
        inventory[label]['last_updated'] = time.time()
        logger.info(f"Removed {label}. Count: {inventory[label]['count']}")
        if inventory[label]['count'] <= 0:
            del inventory[label]

def emit_inventory():                # socket helper
    socketio.emit('inventory_update', {'inventory': inventory,
                                       'timestamp': time.time()})

def encode_frame_to_base64(frame):
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buf).decode('utf-8')

# ----------------------- TRACK-SIDE / VELOCITY -----------------------------
def _velocity(hist):
    if len(hist) < 2: return 0
    dx = hist[-1][0] - hist[0][0]
    dy = hist[-1][1] - hist[0][1]
    dt = hist[-1][2] - hist[0][2]
    return 0 if dt == 0 else math.hypot(dx, dy)/dt

def update_track_side(tid, center, frame=None):
    tr  = tracks[tid]
    old = tr['current_side']
    new = 'below' if center[1] > BOUNDARY_Y else 'above'
    tr['center'] = center
    tr['history'].append((*center, time.time()))
    if len(tr['history']) > TRACK_HISTORY:
        tr['history'].pop(0)
    vel = _velocity(tr['history'])

    if tr['start_side'] is None:
        tr['start_side'] = new
    tr['current_side'] = new

    # crossing into fridge
    if old == 'above' and new == 'below' and not tr['has_crossed_in']:
        if vel > 0:
            _snapshot_and_add(tr, frame)
            tr['has_crossed_in'] = True
            socketio.emit('movement_event',
                          {'type':'input','item':'object','velocity':vel,
                           'timestamp':time.time()})
            emit_inventory()

    # crossing out of fridge
    if old == 'below' and new == 'above' and not tr['has_crossed_out']:
        if vel > 0:
            remove_from_inventory('object')
            tr['has_crossed_out'] = True
            socketio.emit('movement_event',
                          {'type':'output','item':'object','velocity':vel,
                           'timestamp':time.time()})
            emit_inventory()

def _snapshot_and_add(tr, frame):
    """Grab a roomy crop (hand + item), encode to base64, add to inventory."""
    if frame is None:
        add_to_inventory('object')
        return

    # original hand box
    x1, y1, x2, y2 = map(int, tr['box'])
    w  = x2 - x1
    h  = y2 - y1

    # expand symmetrically around center
    cx, cy = x1 + w/2, y1 + h/2
    half_w = max(w * CROP_SCALE / 2, CROP_MIN_PAD)
    half_h = max(h * CROP_SCALE / 2, CROP_MIN_PAD)

    nx1 = int(max(0,     cx - half_w))
    ny1 = int(max(0,     cy - half_h))
    nx2 = int(min(frame.shape[1], cx + half_w))
    ny2 = int(min(frame.shape[0], cy + half_h))

    crop = frame[ny1:ny2, nx1:nx2]
    img64 = encode_frame_to_base64(crop) if crop.size else None
    add_to_inventory('object', img64)


# ----------------------- MAIN VIDEO LOOP -----------------------------------
def generate_frames():
    global tracks
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera"); return

    last_emit = 0.0
    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = digital_zoom(frame, ZOOM_FACTOR)
        h, w  = frame.shape[:2]

        # ---------- Tier-1 : hand boxes ------------------------------------
        detections = []
        for (x1,y1,x2,y2) in hands_in_frame(frame):
            detections.append({'box':(x1,y1,x2,y2), 'conf':1.0})

        # ---------- Tier-2 : YOLO object-ness if no hands ------------------
        if not detections:
            res = yolo_obj(frame, conf=CONF_THRESHOLD,
                           agnostic_nms=True, verbose=False)
            if res and res[0].boxes is not None:
                b = res[0].boxes
                for (x1,y1,x2,y2), conf, cls in zip(b.xyxy.cpu().tolist(),
                                                    b.conf.cpu().tolist(),
                                                    b.cls.int().cpu().tolist()):
                    if cls == PERSON_CLS_ID:        # skip “person”
                        continue
                    detections.append({'box':(x1,y1,x2,y2), 'conf':conf})

        # ---------- Associate detections to existing tracks ----------------
        current_ids = set()
        for det in detections:
            x1,y1,x2,y2 = det['box']
            cx, cy      = (x1+x2)/2, (y1+y2)/2

            # find nearest existing track
            best_id, best_d = None, float('inf')
            for tid,tr in tracks.items():
                d = math.hypot(cx-tr['center'][0], cy-tr['center'][1])
                if d < best_d and d < MAX_TRACK_DISTANCE:
                    best_id, best_d = tid, d

            if best_id is None:                     # create new track
                tid = next(next_track_id)
                tracks[tid] = {'label':'object','center':(cx,cy),
                               'start_side':None,'current_side':None,
                               'has_crossed_in':False,
                               'has_crossed_out':False,'lost_frames':0,
                               'box':(x1,y1,x2,y2),
                               'history':[(cx,cy,time.time())],
                               'confidence':det['conf']}
            else:                                   # update existing
                tid = best_id
                tr  = tracks[tid]
                tr.update({'center':(cx,cy), 'box':(x1,y1,x2,y2),
                           'confidence':det['conf'], 'lost_frames':0})
            current_ids.add(tid)
            update_track_side(tid, (cx,cy), frame)

            # draw box / id
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),
                          (0,255,0),2)
            cv2.putText(frame,f'ID:{tid}',(int(x1),int(y1)-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # ---------- age & purge lost tracks --------------------------------
        for tid in list(tracks):
            if tid not in current_ids:
                tracks[tid]['lost_frames'] += 1
                if tracks[tid]['lost_frames'] > MAX_LOST_FRAMES:
                    del tracks[tid]

        # ---------- draw boundary line & emit counts -----------------------
        cv2.line(frame,(0,BOUNDARY_Y),(w,BOUNDARY_Y),(0,0,255),2)

        now = time.time()
        if now-last_emit >= EMIT_INTERVAL:
            socketio.emit('detection_update',
                           {'items':{'object':len(current_ids)},
                            'timestamp':now})
            last_emit = now

        # ---------- stream frame ------------------------------------------
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
    cap.release()

# ------------------------------- ROUTES ------------------------------------
@app.route('/')
def index():               return render_template('index.html')

@app.route('/video_feed')
def video_feed():          return Response(generate_frames(),
                                   mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- ChatGPT endpoints (unchanged from your script) -----------------
# ... (retain your analyze_image and analyze_inventory endpoints here)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        img_data = request.json.get('image')
        if not img_data:
            return jsonify({'error': 'No image provided'}), 400

        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": [
                    { "type": "text",
                      "text": (
                        "what is in this image? Disregard any text labels "
                        "and identify it yourself. The image contains food "
                        "items that are being tracked entering a fridge. "
                        "Please identify the specific food item and provide "
                        "a confidence level. If it is not food, say unknown."
                      )
                    },
                    { "type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                    }
                ]
            }],
            max_tokens=300
        )

        text = resp.choices[0].message.content
        # parse "apple (90%)" style
        food = text.split('(')[0].strip()
        m = re.search(r'(\d+)%', text)
        conf = int(m.group(1)) if m else 90

        return jsonify({'food': food, 'confidence': conf, 'raw': text})

    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_inventory', methods=['POST'])
def analyze_inventory():
    try:
        data = request.json
        item = data.get('item')
        
        results = {}
        items_to_analyze = [item] if item else inventory.keys()
        
        print("\n=== AI-Verified Inventory ===")
        for item in items_to_analyze:
            if item not in inventory:
                print(f"\n{item}: Not found in inventory")
                continue
                
            item_results = []
            print(f"\n{item.capitalize()}:")
            
            for i, image in enumerate(inventory[item]['images'], 1):
                resp = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{
                        "role": "user",
                        "content": [
                            { "type": "text",
                              "text": (
                                "What is this item? Provide a short, direct answer with just the item name. "
                                "If it's not a food or drink item, respond with 'non-food item'. "
                                "Format: <item name> (confidence: high/medium/low)"
                              )
                            },
                            { "type": "image_url",
                              "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                            }
                        ]
                    }],
                    max_tokens=50
                )

                text = resp.choices[0].message.content
                # Simplify confidence extraction
                if "high" in text.lower():
                    conf = "high"
                elif "medium" in text.lower():
                    conf = "medium"
                else:
                    conf = "low"

                # Extract just the item name without confidence
                food = text.split('(')[0].strip()
                
                print(f"  Image {i}: {food} - Confidence: {conf}")
                item_results.append({
                    'food': food,
                    'confidence': conf,
                    'raw': text
                })
            results[item] = item_results

        print("\n=== End of Verification ===")
        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"Error in analyze_inventory: {e}")
        return jsonify({'error': str(e)}), 500
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
