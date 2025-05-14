#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Smarter fridge‑tracker: curve boundary, multi‑item GPT, hash‑based matching
# ---------------------------------------------------------------------------
import os, time, math, logging, base64, json, itertools, uuid, difflib
import numpy as np, cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO
import mediapipe as mp
from openai import OpenAI

# ------------------------------ CONSTANTS ----------------------------------
VERTEX_Y        = 450
ZOOM_FACTOR     = 1.1
CONF_THRESHOLD  = 0.30
MAX_LOST_FRAMES = 5
MAX_TRACK_DIST  = 200
TRACK_HISTORY   = 10
EMIT_INTERVAL   = 0.5
PERSON_CLS_ID   = 0
CROP_SCALE      = 4
CROP_MIN_PAD    = 100

# ------------------------------ GLOBALS ------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet",
                    logger=False, engineio_logger=False)

yolo_obj = YOLO("yolo11n.pt")                       # fallback detector
mp_hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

tracks          = {}
next_track_id   = itertools.count()
inventory_items = []   # ← our new cache: list of dicts

# --------------------------- SOCKET HELPERS --------------------------------
def emit_inventory():
    """Aggregate cached items and send to UI."""
    agg = {}
    for itm in inventory_items:
        d = agg.setdefault(itm["label"], {"count": 0, "images": []})
        d["count"] += 1
        if itm["image"]:
            d["images"].append(itm["image"])
    socketio.emit("inventory_update", {"inventory": agg, "timestamp": time.time()})

@socketio.on("connect")
def _on_connect(): emit_inventory()

# --------------------------- IMAGE / HASH UTILS ----------------------------
def b64encode_img(bgr):
    _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buf).decode()

def dhash(bgr, size=8):
    bgr = cv2.resize(bgr, (64,64), interpolation=cv2.INTER_AREA)   # cheap downscale
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size+1, size), interpolation=cv2.INTER_AREA)
    diff  = small[:, 1:] > small[:, :-1]
    return sum(1 << i for (i, v) in enumerate(diff.flatten()) if v)

def hamming(a, b):
    return bin(a ^ b).count("1") if a is not None and b is not None else 64

def name_sim(a, b):      # 0‑1 ratio
    return difflib.SequenceMatcher(None, a, b).ratio()

# ------------------------------ CV HELPERS ---------------------------------
def digital_zoom(img, factor):
    h, w = img.shape[:2]; nw, nh = int(w/factor), int(h/factor)
    x1, y1 = (w-nw)//2, (h-nh)//2
    return cv2.resize(img[y1:y1+nh, x1:x1+nw], (w, h), cv2.INTER_LINEAR)

def hands_in_frame(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb); boxes=[]
    if res.multi_hand_landmarks:
        h,w = bgr.shape[:2]
        for hand in res.multi_hand_landmarks:
            xs=[lm.x for lm in hand.landmark]; ys=[lm.y for lm in hand.landmark]
            x1,y1,x2,y2=min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h
            boxes.append((x1,y1,x2,y2))
    return boxes

# ---------------------------- BOUNDARY CURVE -------------------------------
def parabola_y(x, w, h):
    a = 4*(h - VERTEX_Y)/(w**2)
    return VERTEX_Y + a*((x - w/2)**2)

def below_curve(pt, w, h):
    x,y = pt
    return y > parabola_y(x, w, h)

# -------------------------- INVENTORY ACTIONS ------------------------------
def add_item(label, img_bgr=None):
    itm = {"id": uuid.uuid4().hex,
           "label": label,
           "hash": dhash(img_bgr) if img_bgr is not None else None,
           "image": b64encode_img(img_bgr) if img_bgr is not None else None,
           "time": time.time()}
    inventory_items.append(itm)
    logger.info(f"ADD  {label}")
    emit_inventory()

# ---------- INVENTORY HELPERS (direction‑aware) ----------------------------
def add_placeholder(direction, img_bgr, track_hash):
    """direction: 'in' or 'out'.  Placeholders are resolved later by GPT."""
    inventory_items.append({
        "id": uuid.uuid4().hex,
        "direction": direction,        # NEW
        "pending": True,               # until GPT labels it
        "label": "unknown",
        "hash": track_hash,
        "image": b64encode_img(img_bgr),
        "time": time.time()
    })
    emit_inventory()                  # show 'unknown' with gray count


def finalize_in(label, itm):
    """Convert a pending 'in' placeholder into a real stocked item."""
    itm.update({"label": label, "pending": False, "direction": "in"})
    logger.info(f"STORED  {label}")
    emit_inventory()


def finalize_out(label, itm):
    """Remove one stocked item that matches label/hash, then discard exit stub."""
    # find the best stocked match
    best, best_score = None, 0
    for cand in inventory_items:
        if cand["pending"] or cand["direction"] != "in":
            continue
        txt = name_sim(label, cand["label"])
        ham = 1 - hamming(itm["hash"], cand["hash"])/64
        score = max(txt, ham)
        if score > best_score:
            best, best_score = cand, score

    if best_score >= 0.75:
        inventory_items.remove(best)       # remove real item
        inventory_items.remove(itm)        # discard exit stub
        logger.info(f"REMOVED {best['label']} (score {best_score:.2f})")
    else:
        logger.warning(f"No match for exit item '{label}' (score {best_score:.2f})")
        inventory_items.remove(itm)        # drop stub anyway

    emit_inventory()

# ----------------------- TRACK / CROSSING HELPERS --------------------------
def big_crop(box, frame):
    x1,y1,x2,y2 = map(int, box); w,h = x2-x1, y2-y1
    cx,cy = x1+w/2, y1+h/2
    hw = max(w*CROP_SCALE/2, CROP_MIN_PAD); hh = max(h*CROP_SCALE/2, CROP_MIN_PAD)
    nx1,ny1 = int(max(0, cx-hw)), int(max(0, cy-hh))
    nx2,ny2 = int(min(frame.shape[1], cx+hw)), int(min(frame.shape[0], cy+hh))
    return frame[ny1:ny2, nx1:nx2]

def _dy(hist):
    """Signed vertical motion between last two points."""
    if len(hist) < 2:
        return 0
    return hist[-1][1] - hist[-2][1]          # +down, –up

def update_track_side(tid, center, w, h, frame):
    tr = tracks[tid]
    prev_side = tr["side"]
    new_side  = "below" if below_curve(center, w, h) else "above"
    tr["side"] = new_side
    tr["history"].append((*center, time.time()))
    if len(tr["history"]) > TRACK_HISTORY:
        tr["history"].pop(0)

    dy = _dy(tr["history"])

    # entering (above -> below) AND moving down
    if prev_side == "above" and new_side == "below" and dy > 2 and not tr["entered"]:
        crop = big_crop(tr["box"], frame)
        add_placeholder("in", crop, dhash(crop))
        tr["entered"] = True
        tr["exited"]  = False

    # exiting (below -> above) AND moving up
    if prev_side == "below" and new_side == "above" and dy < -2 and not tr["exited"]:
        crop = big_crop(tr["box"], frame)
        add_placeholder("out", crop, dhash(crop))
        tr["exited"]  = True
        tr["entered"] = False


# ----------------------------- MAIN STREAM ---------------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): logger.error("Camera open failed"); return
    last_emit = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = digital_zoom(frame, ZOOM_FACTOR); h,w = frame.shape[:2]

        dets = [{"box":b} for b in hands_in_frame(frame)]
        if dets:
            kept = []
            for cand in dets:
                x1,y1,x2,y2 = cand["box"]
                cx,cy = (x1+x2)/2, (y1+y2)/2
                too_close = False
                for k in kept:
                    kx1,ky1,kx2,ky2 = k["box"]
                    kcx,kcy = (kx1+kx2)/2, (ky1+ky2)/2
                    if math.hypot(cx-kcx, cy-kcy) < 200:   # ≤200 px
                        too_close = True
                        break
                if not too_close:
                    kept.append(cand)
            dets = kept
        if not dets:
            res = yolo_obj(frame, conf=CONF_THRESHOLD, agnostic_nms=True, verbose=False)
            if res and res[0].boxes is not None:
                for (x1,y1,x2,y2),cls in zip(res[0].boxes.xyxy.cpu().tolist(),
                                              res[0].boxes.cls.int().cpu().tolist()):
                    if cls!=PERSON_CLS_ID:
                        dets.append({"box":(x1,y1,x2,y2)})

        current=set()
        for det in dets:
            x1,y1,x2,y2 = det["box"]; cx,cy=(x1+x2)/2,(y1+y2)/2
            best,best_d=None,float('inf')
            for tid,tr in tracks.items():
                d=math.hypot(cx-tr["center"][0], cy-tr["center"][1])
                if d<best_d and d<MAX_TRACK_DIST: best,best_d = tid,d
            if best is None:
                tid = next(next_track_id)
                init_side = "below" if below_curve((cx, cy), w, h) else "above"
                tracks[tid] = {
                    "center":  (cx, cy),
                    "box":     det["box"],
                    "side":    init_side,     # <-- proper initial side
                    "entered": False,
                    "exited":  False,
                    "history": [(cx, cy, time.time())],
                    "lost":    0
                }
            else:
                tid=best; tracks[tid].update({"center":(cx,cy),"box":det["box"],"lost":0})
            current.add(tid)
            update_track_side(tid,(cx,cy),w,h,frame)
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)

        for tid in list(tracks):
            if tid not in current:
                tracks[tid]["lost"] += 1
                if tracks[tid]["lost"]>MAX_LOST_FRAMES: tracks.pop(tid)

        pts=[(x,int(parabola_y(x,w,h))) for x in range(0,w,6)]
        cv2.polylines(frame,[np.array(pts,np.int32)],False,(0,0,255),2)

        now=time.time()
        if now-last_emit>EMIT_INTERVAL:
            socketio.emit("detection_update",
                          {"items":{"object":len(current)}, "timestamp":now})
            last_emit=now

        _, buf = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n')
    cap.release()

# ------------------------------- ROUTES ------------------------------------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed(): return Response(generate_frames(),
    mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------- MULTI‑ITEM GPT ANALYSIS FOR SNAPSHOTS --------------------------
# ---------- MULTI‑ITEM GPT ANALYSIS FOR SNAPSHOTS (fixed) ------------------
@app.route("/analyze_inventory", methods=["POST"])
def analyze_inventory():
    try:
        prompt = ("You are an image inventory assistant. "
                  "Identify EVERY food or drink item you see in the image. "
                  "Ignore hands and background. "
                  "Provide a short, direct answer with just the item name for each detected item. "
                  "If it's not a food or drink item, respond with 'non-food item'. "
                  "Return strict JSON as {\"items\":[{\"name\":\"<item>\",\"confidence\":<0-1>}]}")

        results = {}
        # iterate over copy because we mutate in finalize_xx()
        for itm in list(inventory_items):
            if not itm["pending"]:
                continue

            gpt = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{
                    "role":"user","content":[
                        {"type":"text","text":prompt},
                        {"type":"image_url",
                         "image_url":{"url":f"data:image/jpeg;base64,{itm['image']}"}}]}],
                response_format={"type":"json_object"},
                max_tokens=300)

            parsed = json.loads(gpt.choices[0].message.content)
            if not parsed.get("items"):
                continue

            # use the first detection as canonical
            lbl = parsed["items"][0]["name"].lower().strip()
            conf = round(float(parsed["items"][0]["confidence"])*100)
            results.setdefault(lbl, []).append({
                "food": lbl, "confidence": conf, "image": itm["image"]
            })

            if itm["direction"] == "in":
                finalize_in(lbl, itm)
            else:                                  # 'out'
                finalize_out(lbl, itm)

        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"analyze_inventory error: {e}")
        return jsonify({"error": str(e)}), 500



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
    "item1": {{ "days": num_days_until_item1_expiry, "notes": "specify how to store item1 (eg. store in fridge)" }},
    "item2": {{ "days": num_days_until_item2_expiry, "notes": "specify how to store item2 (eg. keep in cool, dry place)" }},
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
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5001, debug=True)
