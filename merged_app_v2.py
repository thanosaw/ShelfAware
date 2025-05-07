import os, time, math, logging, base64, re, itertools, json
import numpy as np, cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO
import mediapipe as mp
from openai import OpenAI

# ------------------------------ CONSTANTS ----------------------------------
VERTEX_Y              = 450           # y‑coord of parabola vertex (middle top)
ZOOM_FACTOR           = 1.1
CONF_THRESHOLD        = 0.30
MAX_LOST_FRAMES       = 5
MAX_TRACK_DISTANCE    = 200
TRACK_HISTORY         = 10
EMIT_INTERVAL         = 0.5
PERSON_CLS_ID         = 0
CROP_SCALE            = 4
CROP_MIN_PAD          = 100
INVENTORY_UPDATE_INTERVAL = 5.0

# ------------------------------ GLOBALS ------------------------------------
logging.basicConfig(level=logging.INFO)
logger  = logging.getLogger(__name__)

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet',
                    logger=True, engineio_logger=True)

yolo_obj = YOLO("yolo11n.pt")                    # class‑agnostic fallback
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2, model_complexity=0,
    min_detection_confidence=0.35, min_tracking_confidence=0.35)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

inventory, tracks = {}, {}
next_track_id     = itertools.count()

# --------------------------- SOCKET HELPERS --------------------------------
@socketio.on('connect')
def _on_connect():  emit_inventory()
@socketio.on('disconnect')
def _on_disconnect(): pass
@socketio.on('request_inventory')
def _on_req_inv(): emit_inventory()

def emit_inventory():
    socketio.emit('inventory_update',
                  {'inventory': inventory, 'timestamp': time.time()})

# --------------------------- CV HELPERS ------------------------------------
def digital_zoom(f, factor):
    h, w = f.shape[:2]; nw, nh = int(w/factor), int(h/factor)
    x1, y1 = (w-nw)//2, (h-nh)//2
    return cv2.resize(f[y1:y1+nh, x1:x1+nw], (w, h), cv2.INTER_LINEAR)

def hands_in_frame(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    boxes = []
    if res.multi_hand_landmarks:
        h, w = bgr.shape[:2]
        for hand in res.multi_hand_landmarks:
            xs = [lm.x for lm in hand.landmark]; ys = [lm.y for lm in hand.landmark]
            x1, y1, x2, y2 = min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h
            boxes.append((x1, y1, x2, y2))
    return boxes

def encode_frame_to_base64(f):
    _, buf = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY),70])
    return base64.b64encode(buf).decode('utf-8')

def _velocity(hist):
    if len(hist) < 2: return 0
    dx, dy = hist[-1][0]-hist[0][0], hist[-1][1]-hist[0][1]
    dt     = hist[-1][2]-hist[0][2]
    return 0 if dt==0 else math.hypot(dx,dy)/dt

# --------------------------- PARABOLA UTILS --------------------------------
def parabola_y(x, w, h):
    """Return y of boundary parabola at pixel x given frame w,h."""
    a = 4*(h - VERTEX_Y) / (w**2)                # a > 0 (concave‑up in img coords)
    return VERTEX_Y + a*((x - w/2)**2)

def is_below_boundary(pt, w, h):
    """True if (x,y) lies below the parabola."""
    x, y = pt
    return y > parabola_y(x, w, h)

# ----------------------- INVENTORY HELPERS ---------------------------------
def add_to_inventory(label, image=None):
    if label not in inventory:
        inventory[label]={'count':0,'images':[],'last_updated':time.time()}
    inventory[label]['count'] += 1
    if image: inventory[label]['images'].append(image)
    inventory[label]['last_updated']=time.time()

def remove_from_inventory(label):
    if label in inventory and inventory[label]['count']>0:
        inventory[label]['count']-=1
        inventory[label]['last_updated']=time.time()
        if inventory[label]['count']<=0: inventory.pop(label,None)

# ----------------------- SNAPSHOT / TRACK SIDE -----------------------------
def _snapshot_and_add(tr, frame):
    if frame is None: return add_to_inventory('object')
    x1,y1,x2,y2 = map(int,tr['box']); w,h = x2-x1, y2-y1
    cx,cy = x1+w/2, y1+h/2
    hw = max(w*CROP_SCALE/2, CROP_MIN_PAD); hh = max(h*CROP_SCALE/2, CROP_MIN_PAD)
    nx1,ny1 = int(max(0,cx-hw)), int(max(0,cy-hh))
    nx2,ny2 = int(min(frame.shape[1],cx+hw)), int(min(frame.shape[0],cy+hh))
    crop = frame[ny1:ny2, nx1:nx2]
    add_to_inventory('object', encode_frame_to_base64(crop) if crop.size else None)

def update_track_side(tid, center, w, h, frame=None):
    tr = tracks[tid]
    old = tr['current_side']
    new = 'below' if is_below_boundary(center, w, h) else 'above'
    tr['center']=center
    tr['history'].append((*center,time.time()))
    if len(tr['history'])>TRACK_HISTORY: tr['history'].pop(0)
    vel=_velocity(tr['history'])

    if tr['start_side'] is None: tr['start_side']=new
    tr['current_side']=new

    if old=='above' and new=='below' and not tr['has_crossed_in']:
        if vel>0:
            _snapshot_and_add(tr,frame); tr['has_crossed_in']=True
            socketio.emit('movement_event',
                {'type':'input','item':'object','velocity':vel,'timestamp':time.time()})
            emit_inventory()
    if old=='below' and new=='above' and not tr['has_crossed_out']:
        if vel>0:
            remove_from_inventory('object'); tr['has_crossed_out']=True
            socketio.emit('movement_event',
                {'type':'output','item':'object','velocity':vel,'timestamp':time.time()})
            emit_inventory()

# ----------------------- MAIN VIDEO LOOP -----------------------------------
def generate_frames():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): logger.error("Camera open failed"); return
    last_emit=0.0
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=digital_zoom(frame,ZOOM_FACTOR); h,w=frame.shape[:2]

        # ---- detections ---------------------------------------------------
        detections=[{'box':b,'conf':1.0} for b in hands_in_frame(frame)]
        if not detections:                               # fallback YOLO
            res=yolo_obj(frame,conf=CONF_THRESHOLD,agnostic_nms=True,verbose=False)
            if res and res[0].boxes is not None:
                for (x1,y1,x2,y2),conf,cls in zip(res[0].boxes.xyxy.cpu().tolist(),
                                                   res[0].boxes.conf.cpu().tolist(),
                                                   res[0].boxes.cls.int().cpu().tolist()):
                    if cls!=PERSON_CLS_ID:
                        detections.append({'box':(x1,y1,x2,y2),'conf':conf})

        # ---- associate / update tracks -----------------------------------
        current_ids=set()
        for det in detections:
            x1,y1,x2,y2=det['box']; cx,cy=(x1+x2)/2,(y1+y2)/2
            best_id,best_d=None,float('inf')
            for tid,tr in tracks.items():
                d=math.hypot(cx-tr['center'][0],cy-tr['center'][1])
                if d<best_d and d<MAX_TRACK_DISTANCE:
                    best_id,best_d=tid,d
            if best_id is None:
                tid=next(next_track_id)
                tracks[tid]={'label':'object','center':(cx,cy),
                             'start_side':None,'current_side':None,
                             'has_crossed_in':False,'has_crossed_out':False,
                             'lost_frames':0,'box':(x1,y1,x2,y2),
                             'history':[(cx,cy,time.time())],'confidence':det['conf']}
            else:
                tid=best_id; tracks[tid].update(
                    {'center':(cx,cy),'box':(x1,y1,x2,y2),
                     'confidence':det['conf'],'lost_frames':0})
            current_ids.add(tid)
            update_track_side(tid,(cx,cy),w,h,frame)
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(frame,f'ID:{tid}',(int(x1),int(y1)-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # ---- purge lost tracks -------------------------------------------
        for tid in list(tracks):
            if tid not in current_ids:
                tracks[tid]['lost_frames']+=1
                if tracks[tid]['lost_frames']>MAX_LOST_FRAMES:
                    del tracks[tid]

        # ---- draw boundary parabola --------------------------------------
        pts=[(x,int(parabola_y(x,w,h))) for x in range(0,w,8)]
        cv2.polylines(frame,[np.array(pts,np.int32)],False,(0,0,255),2)

        # ---- socket emits -------------------------------------------------
        now=time.time()
        if now-last_emit>=EMIT_INTERVAL:
            socketio.emit('detection_update',
                {'items':{'object':len(current_ids)},'timestamp':now})
            last_emit=now

        # ---- stream multipart -------------------------------------------
        _,buf=cv2.imencode('.jpg',frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n')
    cap.release()

# ------------------------------- ROUTES ------------------------------------
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(),
           mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- ChatGPT endpoints (unchanged from earlier script) -------------
# ... keep your analyze_image, analyze_inventory, get_recipes_and_expirations

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

        return jsonify({'food': food, 'confidence': conf, 'image': img_data, 'raw': text})

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
                                "Before adding a new item to the inventory, check if the snapshot is visually similar (>90%) to any item added in the last 3 seconds. If so ignore it"
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
                    'image': image,
                    'raw': text
                })
            results[item] = item_results

        print("\n=== End of Verification ===")
        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"Error in analyze_inventory: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_recipes_and_expirations', methods=['POST'])
def get_recipes_and_expirations():
    try:
        data = request.json
        food_items = data.get('items', [])
        
        if not food_items:
            return jsonify({'error': 'No food items provided'}), 400
        
        # Format the items for the prompt
        items_list = ", ".join(food_items)
        
        # Create the prompt for OpenAI
        prompt = f"""Based on these food items: {items_list}

1. Suggest 3 recipes that can be made using some or all of these ingredients. For each recipe include:
   - Name
   - Ingredients (indicate which ones are from the provided list)
   - Brief cooking instructions

2. Provide estimated shelf life information for each of these items:
   - For each item, provide approximate days until expiration for a typical fresh item of this type
   - Assume items were fresh when added to inventory

Format your response as valid JSON with this structure:
{{
  "recipes": [
    {{
      "name": "Recipe Name",
      "ingredients": ["ingredient1", "ingredient2", ...],
      "instructions": "Step by step instructions"
    }},
    ...
  ],
  "expirations": {{
    "item1": {{ "days": 5, "notes": "Store in refrigerator" }},
    "item2": {{ "days": 7, "notes": "Keep in cool, dry place" }},
    ...
  }}
}}

IMPORTANT: Ensure your response is ONLY valid JSON that can be parsed, with no additional text."""

        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        result = json.loads(content)
        
        return jsonify(result)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return jsonify({'error': f'Failed to parse OpenAI response: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Error in get_recipes_and_expirations: {e}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5001, debug=True)
