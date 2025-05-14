
#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Fusion fridge‑tracker: hands + YOLO item tracker + GPT identification
# ---------------------------------------------------------------------------
import os, time, math, logging, base64, json, itertools, uuid, difflib
from collections import defaultdict
import numpy as np, cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO
import mediapipe as mp
from openai import OpenAI

# ------------------------------ CONSTANTS ----------------------------------
VERTEX_Y          = 450
ZOOM_FACTOR       = 1.1
CONF_THRESHOLD    = 0.35
MAX_LOST_FRAMES   = 5
MAX_TRACK_DIST    = 200      # hand–hand & item–item association radius
HAND_ITEM_DIST    = 150      # hand–item fusion radius
TRACK_HISTORY     = 10
STABLE_FRAMES     = 3        # min frames before item track is “stable”
EMIT_INTERVAL     = 0.5
PERSON_CLS_ID     = 0
CROP_SCALE        = 4
CROP_MIN_PAD      = 100

# ------------------------------ GLOBALS ------------------------------------
logging.basicConfig(level=logging.INFO)
logger  = logging.getLogger("fridge")

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet",
                    logger=False, engineio_logger=False)

yolo_obj = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                    min_detection_confidence=0.45,
                                    min_tracking_confidence=0.45)
client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# --------------------------- TRACK STATE -----------------------------------
hand_tracks  = {}
item_tracks  = {}
next_hand_id = itertools.count()
next_item_id = itertools.count()
inventory_items = []

# --------------------------- UTILITY FUNCS ---------------------------------
def b64encode_img(bgr):
    _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buf).decode()

def dhash(bgr, size=8):
    bgr = cv2.resize(bgr, (64,64), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size+1, size), interpolation=cv2.INTER_AREA)
    diff  = small[:,1:] > small[:,:-1]
    return sum(1<<i for i,v in enumerate(diff.flatten()) if v)

def name_sim(a,b): return difflib.SequenceMatcher(None,a,b).ratio()
def hamming(a,b):  return bin(a^b).count("1") if a and b else 64

# CV helpers
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

# --------------------------- INVENTORY SOCKET ------------------------------
def emit_inventory():
    agg=defaultdict(lambda:{"count":0,"images":[]})
    for it in inventory_items:
        agg[it["label"]]["count"]+=1
        if it["image"]: agg[it["label"]]["images"].append(it["image"])
    socketio.emit("inventory_update",{"inventory":agg,"timestamp":time.time()})

@socketio.on("connect")
def _on_connect(): emit_inventory()

# --------------------------- BOUNDARY CURVE --------------------------------
def parabola_y(x,w,h): return VERTEX_Y + 4*(h-VERTEX_Y)/(w**2)*(x-w/2)**2
def below_curve(pt,w,h): return pt[1] > parabola_y(pt[0],w,h)

# --------------------------- SNAPSHOT HELPERS ------------------------------
def big_crop(box, frame):
    x1,y1,x2,y2=map(int,box); w,h=x2-x1,y2-y1
    cx,cy=x1+w/2,y1+h/2
    hw=max(w*CROP_SCALE/2,CROP_MIN_PAD);hh=max(h*CROP_SCALE/2,CROP_MIN_PAD)
    nx1,ny1=int(max(0,cx-hw)),int(max(0,cy-hh))
    nx2,ny2=int(min(frame.shape[1],cx+hw)),int(min(frame.shape[0],cy+hh))
    return frame[ny1:ny2,nx1:nx2]

def add_placeholder(direction,img_bgr,track_hash):
    inventory_items.append({
        "id":uuid.uuid4().hex,"label":"unknown","pending":True,
        "direction":direction,"hash":track_hash,"image":b64encode_img(img_bgr),
        "time":time.time()})
    emit_inventory()

def finalize_in(label,itm):
    itm.update({"label":label,"pending":False,"direction":"in"}); emit_inventory()

def finalize_out(label,itm):
    best,best_s=None,0
    for cand in inventory_items:
        if cand["pending"] or cand["direction"]!="in": continue
        s=max(name_sim(label,cand["label"]),1-hamming(itm["hash"],cand["hash"])/64)
        if s>best_s: best,best_s=cand,s
    if best_s>=0.75: inventory_items.remove(best)
    inventory_items.remove(itm)
    emit_inventory()

# --------------------------- DY / MOTION -----------------------------------
def dy(hist): return 0 if len(hist)<2 else hist[-1][1]-hist[-2][1]

# --------------------------- UPDATE HAND TRACKS ----------------------------
def update_hand_side(tid,center,w,h,frame):
    tr=hand_tracks[tid]
    prev=tr["side"]; new="below" if below_curve(center,w,h) else "above"
    tr["side"]=new; tr["hist"].append((*center,time.time()))
    if len(tr["hist"])>TRACK_HISTORY: tr["hist"].pop(0)
    v=dy(tr["hist"])

    # ----- crossing detection --------------------------------------------
    if prev=="above" and new=="below" and v>2 and not tr["flag"]:
        # look for nearby stable item track
        near=[it for it in item_tracks.values()
              if math.hypot(center[0]-it["center"][0], center[1]-it["center"][1])<HAND_ITEM_DIST
              and it["stable"]]
        if near:
            crop=big_crop(near[0]["box"],frame)
            add_placeholder("in",crop,dhash(crop))
            tr["flag"]=True
    if prev=="below" and new=="above" and v<-2 and not tr["flag"]:
        near=[it for it in item_tracks.values()
              if math.hypot(center[0]-it["center"][0], center[1]-it["center"][1])<HAND_ITEM_DIST
              and it["stable"]]
        if near:
            crop=big_crop(near[0]["box"],frame)
            add_placeholder("out",crop,dhash(crop))
            tr["flag"]=True
    # reset flag when object returns to original side
    if prev!=new and abs(v)<1: tr["flag"]=False

# --------------------------- UPDATE ITEM TRACKS ---------------------------
def update_item_side(tid,center,w,h,frame):
    tr=item_tracks[tid]
    prev=tr["side"]; new="below" if below_curve(center,w,h) else "above"
    if tr["stable"]:
        tr["side"]=new
    tr["hist"].append((*center,time.time()))
    if len(tr["hist"])>STABLE_FRAMES: tr["hist"].pop(0)
    if len(tr["hist"])==STABLE_FRAMES: tr["stable"]=True
    v=dy(tr["hist"])

    # item crosses alone
    if prev=="above" and new=="below" and v>2 and tr["stable"] and not tr["flag"]:
        crop=big_crop(tr["box"],frame)
        add_placeholder("in",crop,dhash(crop)); tr["flag"]=True
    if prev=="below" and new=="above" and v<-2 and tr["stable"] and not tr["flag"]:
        crop=big_crop(tr["box"],frame)
        add_placeholder("out",crop,dhash(crop)); tr["flag"]=True

# --------------------------- MAIN VIDEO LOOP -------------------------------
def generate_frames():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): logger.error("cam?"); return
    last_emit=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)                  # mirror
        # frame=cv2.resize(frame,(960,540))
        frame_zoom=digital_zoom(frame,ZOOM_FACTOR)
        h,w=frame_zoom.shape[:2]

        # ----------------- 1. detect hands --------------------------------
        hand_dets=[{"box":b} for b in hands_in_frame(frame_zoom)]
        # remove dup hand boxes within 200 px
        dedup=[]
        for d in hand_dets:
            cx,cy=((d["box"][0]+d["box"][2])/2,(d["box"][1]+d["box"][3])/2)
            if all(math.hypot(cx-(e["cx"]),cy-(e["cy"]))>200 for e in dedup):
                d["cx"],d["cy"]=cx,cy; dedup.append(d)
        hand_dets=dedup

        # ----------------- 2. detect all items via YOLO -------------------
        item_dets=[]
        yolo=yolo_obj(frame_zoom,conf=CONF_THRESHOLD,agnostic_nms=True,verbose=False)
        if yolo and yolo[0].boxes is not None:
            for (x1,y1,x2,y2),cls in zip(yolo[0].boxes.xyxy.cpu().tolist(),
                                         yolo[0].boxes.cls.int().cpu().tolist()):
                if cls!=PERSON_CLS_ID:
                    item_dets.append({"box":(x1,y1,x2,y2)})

        # ----------------- 3. update hand tracks --------------------------
        cur_h=set()
        for det in hand_dets:
            cx,cy=det["cx"],det["cy"]
            best,best_d=None,float('inf')
            for tid,t in hand_tracks.items():
                d=math.hypot(cx-t["center"][0], cy-t["center"][1])
                if d<best_d and d<MAX_TRACK_DIST: best,best_d=tid,d
            if best is None:
                tid=next(next_hand_id)
                init="below" if below_curve((cx,cy),w,h) else "above"
                hand_tracks[tid]={"center":(cx,cy),"box":det["box"],
                                  "side":init,"hist":[(cx,cy,time.time())],
                                  "flag":False,"lost":0}
            else:
                tid=best; hand_tracks[tid].update(center=(cx,cy),box=det["box"],lost=0)
            cur_h.add(tid); update_hand_side(tid,(cx,cy),w,h,frame_zoom)
            cv2.rectangle(frame_zoom,(int(det["box"][0]),int(det["box"][1])),
                          (int(det["box"][2]),int(det["box"][3])),(0,255,0),2)

        # purge lost hand tracks
        for tid in list(hand_tracks):
            if tid not in cur_h:
                hand_tracks[tid]["lost"]+=1
                if hand_tracks[tid]["lost"]>MAX_LOST_FRAMES:
                    hand_tracks.pop(tid,None)

        # ----------------- 4. update item tracks --------------------------
        cur_i=set()
        for det in item_dets:
            x1,y1,x2,y2=det["box"]
            cx,cy=(x1+x2)/2,(y1+y2)/2
            best,best_d=None,float('inf')
            for iid,it in item_tracks.items():
                d=math.hypot(cx-it["center"][0], cy-it["center"][1])
                if d<best_d and d<MAX_TRACK_DIST: best,best_d=iid,d
            if best is None:
                iid=next(next_item_id)
                init="below" if below_curve((cx,cy),w,h) else "above"
                item_tracks[iid]={"center":(cx,cy),"box":det["box"],
                                  "side":init,"hist":[(cx,cy,time.time())],
                                  "stable":False,"flag":False,"lost":0}
            else:
                iid=best; item_tracks[iid].update(center=(cx,cy),box=det["box"],lost=0)
            cur_i.add(iid); update_item_side(iid,(cx,cy),w,h,frame_zoom)
            cv2.rectangle(frame_zoom,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)

        # purge lost item tracks
        for iid in list(item_tracks):
            if iid not in cur_i:
                item_tracks[iid]["lost"]+=1
                if item_tracks[iid]["lost"]>MAX_LOST_FRAMES:
                    item_tracks.pop(iid,None)

        # draw boundary
        pts=[(x,int(parabola_y(x,w,h))) for x in range(0,w,8)]
        cv2.polylines(frame_zoom,[np.array(pts,np.int32)],False,(0,0,255),2)

        # emit raw detection count (debug)
        now=time.time()
        if now-last_emit>EMIT_INTERVAL:
            socketio.emit("detection_update",
                          {"items":{"hand":len(cur_h),"item":len(cur_i)},
                           "timestamp":now})
            last_emit=now

        _,buf=cv2.imencode(".jpg",frame_zoom)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n')
    cap.release()

# ------------------------------- FLASK ROUTES ------------------------------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed(): return Response(generate_frames(),
    mimetype="multipart/x-mixed-replace; boundary=frame")

# -------------- GPT RESOLUTION ENDPOINT (unchanged wrt fusion) ------------
@app.route("/analyze_inventory", methods=["POST"])
def analyze_inventory():
    try:
        prompt=("You are an image inventory assistant. Identify EVERY food or drink "
                "item you see in the image. Ignore hands and background. Provide "
                "a short, direct answer with just the item name for each detected "
                "item. If it's not a food or drink item, respond with 'non-food item'. "
                "Return strict JSON as {\"items\":[{\"name\":\"<item>\",\"confidence\":<0-1>}]}")

        results={}
        for itm in list(inventory_items):
            if not itm["pending"]: continue
            gpt=client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role":"user","content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{itm['image']}"}}]}],
                response_format={"type":"json_object"},max_tokens=300)
            parsed=json.loads(gpt.choices[0].message.content)
            if not parsed.get("items"): continue
            lbl=parsed["items"][0]["name"].lower().strip()
            conf=round(float(parsed["items"][0]["confidence"])*100)
            results.setdefault(lbl,[]).append({"food":lbl,"confidence":conf,"image":itm["image"]})
            if itm["direction"]=="in":  finalize_in(lbl,itm)
            else:                       finalize_out(lbl,itm)
        return jsonify({"results":results})
    except Exception as e:
        logger.error(e); return jsonify({"error":str(e)}),500

# -- get_recipes_and_expirations unchanged (omitted for brevity) ------------
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
    

# ---------- CLEAR INVENTORY -------------------------------------------------
@app.route("/clear_inventory", methods=["POST"])
def clear_inventory():
    """Wipe both stocked and pending items, then notify the UI."""
    inventory_items.clear()          # forget everything
    emit_inventory()                 # push empty current inventory
    socketio.emit("ai_inventory_cleared")   # tell UI to hide AI panel
    return jsonify({"status": "cleared"})

# ---------------------------------------------------------------------------
if __name__=="__main__":
    socketio.run(app, host="127.0.0.1", port=5001, debug=True)
