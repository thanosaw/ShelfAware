#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Fusion fridge‑tracker: hands + YOLO item tracker + GPT identification
# ---------------------------------------------------------------------------
import os, time, math, logging, base64, json, itertools, uuid, difflib, re
from collections import defaultdict
import numpy as np, cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO
import mediapipe as mp
from openai import OpenAI
import subprocess
from dotenv import load_dotenv
from PIL import Image
import pytesseract, io, base64


# Load environment variables
load_dotenv()

# ------------------------------ CONSTANTS ----------------------------------
VERTEX_Y          = 450
ZOOM_FACTOR       = 1.0
CONF_THRESHOLD    = 0.35
MAX_LOST_FRAMES   = 10
MAX_TRACK_DIST    = 400      # hand–hand & item–item association radius
HAND_ITEM_DIST    = 250      # hand–item fusion radius
TRACK_HISTORY     = 10
STABLE_FRAMES     = 1        # min frames before item track is "stable"
EMIT_INTERVAL     = 0.5
PERSON_CLS_ID     = 0
CROP_SCALE        = 6
CROP_MIN_PAD      = 150
# --- confidence gates -------------------------------------------------
NEAR_HAND_DIST   = 120      # px – centre‑to‑centre to call it "in hand"
CONF_NEAR_HAND   = 0.25     # accept weak box if it's near a hand
CONF_SOLO_OBJECT = 0.4     # stricter when object crosses alone

# Sound file paths
ADD_SOUND = "sounds/add_item.mp3"
REMOVE_SOUND = "sounds/remove_item.mp3"

def play_sound(sound_file):
    """Play a sound file using afplay (macOS) in a non-blocking way"""
    try:
        subprocess.Popen(['afplay', sound_file])
    except Exception as e:
        logger.error(f"Failed to play sound {sound_file}: {e}")

# ------------------------------ GLOBALS ------------------------------------
logging.basicConfig(level=logging.INFO)
logger  = logging.getLogger("fridge")

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet",
                    logger=False, engineio_logger=False)

yolo_obj = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
assert pytesseract.pytesseract.tesseract_cmd, "TESSERACT_CMD is not set!"

# --------------------------- TRACK STATE -----------------------------------
hand_tracks  = {}
item_tracks  = {}
next_hand_id = itertools.count()
next_item_id = itertools.count()
inventory_items = []  # Initialize as empty list

# Add this near the top with other globals
detection_log = []

def update_detection_log(action, item_name, count=1, confidence=None):
    """Update the detection log with item actions"""
    timestamp = time.time()
    log_entry = {
        "timestamp": timestamp,
        "action": action,  # "added" or "removed"
        "item": item_name,
        "count": count,
        "confidence": confidence
    }
    detection_log.append(log_entry)
    # Keep only last 100 entries
    if len(detection_log) > 100:
        detection_log.pop(0)
    # Emit the update to all clients
    socketio.emit("detection_update", {
        "items": {item_name: count},
        "action": action,
        "timestamp": timestamp
    })

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

def normalize_food_name(name):
    """Normalize food names for better matching"""
    if not name:
        return ""
    # Convert to lowercase
    name = name.lower()
    
    # Remove parenthetical descriptions
    name = re.sub(r'\([^)]*\)', '', name).strip()
    
    # Remove common prefixes/suffixes
    prefixes = ["a ", "an ", "the ", "some ", "fresh ", "whole ", "organic "]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Remove common suffixes
    suffixes = ["s", "es", "ies"]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Handle container types
    containers = {
        "bottle": ["bottle of", "bottled", "bottles"],
        "can": ["can of", "canned", "cans"],
        "jar": ["jar of", "jars"],
        "box": ["box of", "boxes"],
        "pack": ["pack of", "package of", "packages"],
        "container": ["container of", "containers"]
    }
    
    for container, variants in containers.items():
        for variant in variants:
            if variant in name:
                name = name.replace(variant, container)
    
    # Handle beverage types
    beverages = {
        "water": ["sparkling water", "mineral water", "spring water", "drinking water"],
        "soda": ["carbonated drink", "soft drink", "pop", "cola"],
        "juice": ["fruit juice", "orange juice", "apple juice"],
        "milk": ["dairy milk", "almond milk", "soy milk"],
        "beer": ["ale", "lager", "brew"],
        "wine": ["red wine", "white wine", "rose wine"]
    }
    
    for base, variants in beverages.items():
        for variant in variants:
            if variant in name:
                name = name.replace(variant, base)
    
    # Replace common variations
    variations = {
        "tomato": "tomatoes",
        "potato": "potatoes",
        "apple": "apples",
        "banana": "bananas",
        "orange": "oranges",
        "milk": "dairy milk",
        "cheese": "cheese",
        "yogurt": "yogurt",
        "bread": "bread",
        "egg": "eggs",
        "chicken": "chicken",
        "beef": "beef",
        "pork": "pork",
        "fish": "fish",
        "rice": "rice",
        "pasta": "pasta",
        "sauce": "sauce",
        "juice": "juice",
        "water": "water",
        "soda": "soda",
        "beer": "beer",
        "wine": "wine"
    }
    
    for base, variant in variations.items():
        if name == base or name == variant:
            return base
    
    # Clean up any remaining whitespace
    name = ' '.join(name.split())
    return name

def levenshtein(a, b):
    """Calculate the Levenshtein distance between two strings.
    This measures how many single-character edits are needed to change one string into another.
    """
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    
    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def name_sim(a, b):
    """Calculate similarity between two food names"""
    # Normalize both names
    a = normalize_food_name(a)
    b = normalize_food_name(b)
    
    # If either name is empty after normalization, return 0
    if not a or not b:
        return 0.0
    
    # Direct match after normalization
    if a == b:
        return 1.0
    
    # Check if one name contains the other
    if a in b or b in a:
        return 0.9
    
    # Split into words and check for partial matches
    a_words = set(a.split())
    b_words = set(b.split())
    common_words = a_words.intersection(b_words)
    
    if common_words:
        # Calculate word overlap score
        word_overlap = len(common_words) / max(len(a_words), len(b_words))
        if word_overlap > 0.5:  # If more than half the words match
            return 0.8
    
    # Calculate Levenshtein distance
    distance = levenshtein(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    
    # Convert distance to similarity score
    similarity = 1.0 - (distance / max_len)
    
    # Boost similarity for common food name patterns
    if similarity > 0.5:  # Lowered threshold to catch more potential matches
        # Check for common food name patterns
        common_patterns = [
            ("whole", "regular"),
            ("organic", "regular"),
            ("fresh", ""),
            ("frozen", ""),
            ("canned", ""),
            ("dried", ""),
            ("raw", ""),
            ("cooked", ""),
            ("ripe", ""),
            ("unripe", ""),
            ("bottle", "container"),
            ("can", "container"),
            ("jar", "container"),
            ("box", "container"),
            ("pack", "container")
        ]
        for pattern1, pattern2 in common_patterns:
            if (pattern1 in a and pattern2 in b) or (pattern2 in a and pattern1 in b):
                similarity += 0.3  # Increased boost for container matches
                break
    
    return min(1.0, similarity)

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
def bulk_add(label:str, qty:int, confidence:float=100.0, expiration_days:int=None):
    """Add *qty* copies of *label* (already lowercase) with metadata."""
    global inventory_items  # Ensure we're using the global variable
    if not isinstance(inventory_items, list):
        inventory_items = []  # Reset if somehow corrupted
        
    for _ in range(max(1, qty)):
        inventory_items.append({
            "id": uuid.uuid4().hex,
            "label": label,
            "pending": False,
            "direction": "in",
            "time": time.time(),
            "image": None,
            "confidence": confidence,
            "expiration_days": expiration_days,
            "added_timestamp": time.time(),
            "last_updated": time.time()
        })

def emit_inventory(source=None):
    """Aggregate items then push to every client."""
    global inventory_items  # Ensure we're using the global variable
    if not isinstance(inventory_items, list):
        inventory_items = []  # Reset if somehow corrupted
        
    agg = defaultdict(lambda: {
        "count": 0, 
        "images": [],
        "confidence": 0,
        "expiration_days": None,
        "added_timestamp": None,
        "last_updated": None
    })
    
    for it in inventory_items:
        agg[it["label"]]["count"] += 1
        if it["image"]:
            agg[it["label"]]["images"].append(it["image"])
        # Update metadata with the most recent values
        agg[it["label"]]["confidence"] = max(agg[it["label"]]["confidence"], it.get("confidence", 0))
        agg[it["label"]]["expiration_days"] = it.get("expiration_days")
        agg[it["label"]]["added_timestamp"] = it.get("added_timestamp")
        agg[it["label"]]["last_updated"] = it.get("last_updated")

    socketio.emit("inventory_update",
                  {"inventory": agg,
                   "timestamp": time.time(),
                   "source": source})

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
    itm = {
        "id": uuid.uuid4().hex,
        "label": "unknown",
        "pending": True,
        "direction": direction,
        "hash": track_hash,
        "image": b64encode_img(img_bgr),
        "time": time.time(),
        "confidence": 0,
        "expiration_days": None,
        "added_timestamp": time.time(),
        "last_updated": time.time()
    }
    inventory_items.append(itm)
    emit_inventory()
    # Automatically trigger GPT analysis in the background
    socketio.start_background_task(process_pending_item, itm)

def finalize_in(label,itm):
    itm.update({
        "label": label,
        "pending": False,
        "direction": "in",
        "last_updated": time.time()
    })
    # Log the addition in detection log
    update_detection_log("added", label, 1, itm.get('confidence'))
    emit_inventory()

def get_gpt_similarity(item_name, inventory_items):
    """Query GPT to find the most similar item in inventory"""
    try:
        # Create a list of inventory items for comparison
        inventory_names = [itm["label"] for itm in inventory_items if not itm["pending"] and itm["direction"] == "in"]
        if not inventory_names:
            return None, 0.0

        # Create the prompt for GPT
        prompt = f"""Given a food item "{item_name}", which item from this list is most similar to it? Ensure the items can reasonably be considered the same item, not just related to each other. If no matches make sense, indicate so.
List of items: {', '.join(inventory_names)}

Return your response as a JSON object with this exact structure:
{{
    "most_similar": "item_name",
    "confidence": 0.95,
    "reason": "brief explanation"
}}

Only return the JSON object, no other text. The confidence should be between 0 and 1."""

        # Log the query
        logger.info(f"GPT Similarity Query - Item: {item_name}")
        logger.info(f"GPT Similarity Query - Inventory: {inventory_names}")
        logger.info(f"GPT Similarity Query - Full prompt: {prompt}")

        # Query GPT
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=150
        )

        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Log the response
        logger.info(f"GPT Similarity Response - Raw: {response.choices[0].message.content}")
        logger.info(f"GPT Similarity Response - Parsed: {result}")
        
        return result["most_similar"], result["confidence"]

    except Exception as e:
        logger.error(f"Error in GPT similarity matching: {e}")
        return None, 0.0

def finalize_out(label,itm):
    """Handle item removal from the fridge inventory"""
    try:
        logger.info(f"Attempting to remove item: {label}")
        
        # First, ensure we have a valid label
        if not label or label.lower() == "unknown":
            logger.warning("Invalid or unknown label for removal")
            inventory_items.remove(itm)
            emit_inventory(source="removed")
            return

        # Find all matching items in inventory (case-insensitive)
        matching_items = []
        normalized_label = normalize_food_name(label)
        logger.info(f"Normalized label: {normalized_label}")
        
        for cand in inventory_items:
            if not cand["pending"] and cand["direction"] == "in":
                normalized_cand = normalize_food_name(cand["label"])
                logger.info(f"Comparing with normalized item: {normalized_cand}")
                if normalized_cand == normalized_label:
                    matching_items.append(cand)
                    logger.info(f"Found exact match after normalization: {cand['label']}")

        if matching_items:
            # Remove the most recently added item (last in the list)
            item_to_remove = matching_items[-1]
            logger.info(f"Removing item from inventory: {item_to_remove['label']}")
            inventory_items.remove(item_to_remove)
            update_detection_log("removed", item_to_remove['label'], 1, item_to_remove.get('confidence'))
        else:
            # If no exact match, try GPT-based matching
            best_match, confidence = get_gpt_similarity(label, inventory_items)
            
            if best_match and confidence >= 0.80:  # Using 0.80 as threshold
                # Find the matching item in inventory
                for cand in inventory_items:
                    if cand["label"].lower() == best_match.lower() and not cand["pending"] and cand["direction"] == "in":
                        logger.info(f"Removing GPT-matched item: {cand['label']} (confidence: {confidence:.2f})")
                        inventory_items.remove(cand)
                        update_detection_log("removed", cand['label'], 1, cand.get('confidence'))
                        break
            else:
                logger.warning(f"No good match found for removal of {label}")

        # Always remove the temporary detection item
        if itm in inventory_items:
            inventory_items.remove(itm)
            logger.info("Removed temporary detection item")
        
        # Emit the updated inventory
        emit_inventory(source="removed")
        logger.info("Inventory updated after removal")
        
    except Exception as e:
        logger.error(f"Error in finalize_out: {e}")
        # Ensure we at least remove the temporary detection item
        try:
            if itm in inventory_items:
                inventory_items.remove(itm)
                emit_inventory(source="removed")
                logger.info("Removed temporary item after error")
        except Exception as e2:
            logger.error(f"Failed to remove temporary item: {e2}")

def process_pending_item(itm):
    """Run GPT analysis on a single pending inventory item."""
    try:
        if not itm.get("image"):
            return None

        prompt = (
            "You are an image inventory assistant. Identify EVERY food or drink "
            "item you see in the image. Ignore hands and background. Provide "
            "a short, direct answer with just the item name for each detected "
            "item. If it's not a food or drink item, respond with 'non-food item'. "
            "Return strict JSON as {\"items\":[{\"name\":\"<item>\",\"confidence\":<0-1>}]}"
        )

        gpt = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{itm['image']}"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
        )

        parsed = json.loads(gpt.choices[0].message.content)
        if not parsed.get("items"):
            return None

        lbl = parsed["items"][0]["name"].lower().strip()
        conf = round(float(parsed["items"][0]["confidence"])*100)

        if lbl == "non-food item":
            if itm in inventory_items:
                inventory_items.remove(itm)
                emit_inventory()
            return None

        expiration_prompt = (
            f"What is the typical shelf life in days for {lbl} when stored properly? Return only a number."
        )
        expiration_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": expiration_prompt}],
            max_tokens=10,
        )
        try:
            expiration_days = int(expiration_response.choices[0].message.content.strip())
        except (ValueError, AttributeError):
            expiration_days = None

        result = {
            "food": lbl,
            "confidence": conf,
            "image": itm.get("image"),
            "expiration_days": expiration_days,
        }

        if itm.get("direction") == "in":
            finalize_in(lbl, itm)
        else:
            finalize_out(lbl, itm)

        itm.update(
            {
                "confidence": conf,
                "expiration_days": expiration_days,
                "last_updated": time.time(),
            }
        )

        return result

    except Exception as e:
        logger.error(f"Error processing pending item: {e}")
        return None

# --------------------------- DY / MOTION -----------------------------------
def dy(hist): return 0 if len(hist)<2 else hist[-1][1]-hist[-2][1]

# --------------------------- UPDATE HAND TRACKS ----------------------------
# ────────────── UPDATE THIS WHOLE FUNCTION ───────────────────────────────
def update_hand_side(tid, center, w, h, frame):
    tr   = hand_tracks[tid]
    prev = tr["side"]
    new  = "below" if below_curve(center, w, h) else "above"

    tr["side"] = new
    tr["hist"].append((*center, time.time()))
    if len(tr["hist"]) > TRACK_HISTORY:
        tr["hist"].pop(0)

    v = dy(tr["hist"])          # positive = moving down, negative = moving up

    # ---------------------------------------------------------------------
    # 1. HAND GOES *IN*  (crosses from above → below)
    # ---------------------------------------------------------------------
    if prev == "above" and new == "below" and v > 2 and not tr["flag"]:
        crop = big_crop(tr["box"], frame)           # crop around the hand box
        play_sound(ADD_SOUND)
        add_placeholder("in", crop, dhash(crop))
        tr["flag"] = True                           # prevent double‑fire until reset

    # ---------------------------------------------------------------------
    # 2. HAND COMES *OUT* (crosses from below → above)
    # ---------------------------------------------------------------------
    elif prev == "below" and new == "above" and v < -2 and not tr["flag"]:
        crop = big_crop(tr["box"], frame)
        play_sound(REMOVE_SOUND)
        add_placeholder("out", crop, dhash(crop))
        tr["flag"] = True

    # ---------------------------------------------------------------------
    # 3. RESET the one‑shot flag once the hand returns to its original side
    #    (prevents continuous firing while the hand is on the same side)
    # ---------------------------------------------------------------------
    if prev != new and abs(v) < 1:
        tr["flag"] = False


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

    # item crosses alone - only process if not already processed
    if prev=="above" and new=="below" and v>2 and not tr["flag"] and not tr.get("processed", False):
        crop=big_crop(tr["box"],frame)
        play_sound(ADD_SOUND)  # Play sound when we crop
        add_placeholder("in",crop,dhash(crop))
        tr["flag"]=True
        tr["processed"] = True
    if prev=="below" and new=="above" and v<-2 and not tr["flag"] and not tr.get("processed", False):
        crop=big_crop(tr["box"],frame)
        play_sound(REMOVE_SOUND)  # Play sound when we crop
        add_placeholder("out",crop,dhash(crop))
        tr["flag"]=True
        tr["processed"] = True

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
        # remove dup hand boxes within 200 px
        dedup=[]
        for d in hand_dets:
            cx,cy=((d["box"][0]+d["box"][2])/2,(d["box"][1]+d["box"][3])/2)
            if all(math.hypot(cx-(e["cx"]),cy-(e["cy"]))>200 for e in dedup):
                d["cx"],d["cy"]=cx,cy; dedup.append(d)
        hand_dets=dedup

        # ----------------- 2. detect all items via YOLO -------------------
        hand_centres = [(d["cx"], d["cy"]) for d in hand_dets]   # for later test


        # --------------- 2.5  YOLO object pass with adaptive thresholds ----------
        item_dets = []

        # run once with the *lowest* threshold so nothing is missed
        yolo = yolo_obj(frame_zoom,
                        conf=min(CONF_NEAR_HAND, CONF_SOLO_OBJECT),
                        agnostic_nms=True,
                        verbose=False)

        if yolo and yolo[0].boxes is not None:
            for (x1,y1,x2,y2), conf, cls in zip(
                    yolo[0].boxes.xyxy.cpu().tolist(),
                    yolo[0].boxes.conf.cpu().tolist(),
                    yolo[0].boxes.cls.int().cpu().tolist()):

                if cls == PERSON_CLS_ID:          # skip people
                    continue

                # decide which gate applies
                cx, cy = (x1+x2)/2, (y1+y2)/2
                near = any(math.hypot(cx-hx, cy-hy) <= NEAR_HAND_DIST
                        for (hx,hy) in hand_centres)

                thresh = CONF_NEAR_HAND if near else CONF_SOLO_OBJECT
                if conf >= thresh:
                    item_dets.append({"box": (x1, y1, x2, y2)})


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
                                  "stable":False,"flag":False,"lost":0,
                                  "processed":False}  # Add processed flag
            else:
                iid=best; item_tracks[iid].update(center=(cx,cy),box=det["box"],lost=0)
            cur_i.add(iid); update_item_side(iid,(cx,cy),w,h,frame_zoom)
            cv2.rectangle(frame_zoom,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)

        # purge lost item tracks and reset processed flag
        for iid in list(item_tracks):
            if iid not in cur_i:
                item_tracks[iid]["lost"]+=1
                if item_tracks[iid]["lost"]>MAX_LOST_FRAMES:
                    item_tracks.pop(iid,None)
            else:
                # Reset processed flag when item is no longer near hands
                if not any(math.hypot(item_tracks[iid]["center"][0]-hx, 
                                    item_tracks[iid]["center"][1]-hy) <= HAND_ITEM_DIST 
                          for (hx,hy) in hand_centres):
                    item_tracks[iid]["processed"] = False

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

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        img_data = request.json.get('image')
        if not img_data:
            return jsonify({'error': 'No image provided'}), 400

        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text',
                     'text': (
                         'what is in this image? Disregard any text labels and '
                         'identify it yourself. The image contains food items '
                         'that are being tracked entering a fridge. Please '
                         'identify the specific food item and provide a '
                         'confidence level. If it is not food, say unknown.'
                     )},
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/jpeg;base64,{img_data}'}}
                ]
            }],
            max_tokens=300
        )

        text = resp.choices[0].message.content
        food = text.split('(')[0].strip()
        m = re.search(r'(\d+)%', text)
        conf = int(m.group(1)) if m else 90

        return jsonify({'food': food, 'confidence': conf, 'raw': text})

    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return jsonify({'error': str(e)}), 500

# -------------- GPT RESOLUTION ENDPOINT (unchanged wrt fusion) ------------
@app.route("/analyze_inventory", methods=["POST"])
def analyze_inventory():
    try:
        global inventory_items  # Ensure we're using the global variable
        if not isinstance(inventory_items, list):
            inventory_items = []  # Reset if somehow corrupted
            
        prompt=("You are an image inventory assistant. Identify EVERY food or drink "
                "item you see in the image. Ignore hands and background. Provide "
                "a short, direct answer with just the item name for each detected "
                "item. If it's not a food or drink item, respond with 'non-food item'. "
                "Return strict JSON as {\"items\":[{\"name\":\"<item>\",\"confidence\":<0-1>}]}")

        results = {}
        pending_items = [itm for itm in inventory_items if itm.get("pending", False)]

        if not pending_items:
            pending_items = [itm for itm in inventory_items if not itm.get("pending", False)]
            if not pending_items:
                return jsonify({"results": {}})

        for itm in pending_items:
            result = process_pending_item(itm)
            if result:
                lbl = result["food"]
                results.setdefault(lbl, []).append(result)

        return jsonify({"results":results})
    except Exception as e:
        logger.error(f"Error in analyze_inventory: {e}")
        return jsonify({"error":str(e)}),500

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
            model="gpt-4.1",  # Using GPT-4 for better recipe generation
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Validate the response structure
        if not isinstance(result, dict) or 'recipes' not in result:
            raise ValueError("Invalid response format from OpenAI")
            
        # Ensure we have at least one recipe
        if not result['recipes'] or not isinstance(result['recipes'], list):
            result['recipes'] = []
            
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
    try:
        inventory_items.clear()  # forget everything
        emit_inventory(source="clear")  # push empty current inventory
        socketio.emit('ai_inventory_cleared')
        return jsonify({"status": "cleared"})
    except Exception as e:
        logger.error(f"Error clearing inventory: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ---------- RECEIPT OCR & INVENTORY UPDATE ---------------------------------
@app.route("/upload_receipt", methods=["POST"])
def upload_receipt():
    """
    Body  : { "image" : "<base64‑jpeg‑or‑png>" }
    Return: { "items": [ {"name":"milk", "qty":2}, ... ] }
    """
    try:
        b64 = request.json.get("image")
        if not b64:
            return jsonify({"error": "No image"}), 400

        if not b64 or "base64," not in b64:
            return jsonify({"error": "Invalid or missing base64 image"}), 400

        img_bytes = base64.b64decode(b64.split("base64,")[-1])
        img = Image.open(io.BytesIO(img_bytes))
        raw_text = pytesseract.image_to_string(img)

        # Ask GPT to turn messy OCR into structured grocery lines
        prompt = (
            "Below is raw OCR text from a grocery receipt.\n"
            "Extract a JSON array called 'items' where each entry has "
            "'name' (lower‑case, no brand codes) and 'qty' (integer, "
            "default 1 if missing).  Only include food and drink items. "
            "Ignore non-food items, prices, totals, loyalty text, "
            "coupons, taxes, etc.\n\nOCR:\n```" + raw_text + "```"
        )

        gpt = client.chat.completions.create(
            model="gpt-4o-mini",              # fast & cheap, or gpt‑3.5‑turbo
            messages=[{"role":"user","content":prompt}],
            response_format={"type": "json_object"},
            max_tokens=400
        )
        parsed = json.loads(gpt.choices[0].message.content)
        items  = parsed.get("items", [])

        # update inventory
        for it in items:
            bulk_add(it["name"].lower().strip(), int(it.get("qty",1)))

        emit_inventory(source="receipt")
        return jsonify({"items": items})

    except Exception as e:
        logger.error(f"receipt upload: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/update_inventory_count', methods=['POST'])
def update_inventory_count():
    try:
        data = request.json
        item = data.get('item', '').lower()  # Convert to lowercase
        delta = data.get('delta', 0)
        
        if not item:
            return jsonify({'error': 'Item name is required'}), 400
            
        # Find all matching items in inventory (case-insensitive)
        matching_items = [itm for itm in inventory_items if itm["label"].lower() == item]
        
        # Count non-pending items
        new_count = 0
        for itm in matching_items:
            if not itm.get("pending", False):  # Only count non-pending items
                new_count += 1
        
        # Apply delta
        if delta > 0:
            # Add new items
            for _ in range(delta):
                inventory_items.append({
                    "id": uuid.uuid4().hex,
                    "label": item,
                    "pending": False,
                    "direction": "in",
                    "time": time.time(),
                    "image": None,
                    "confidence": 100.0,  # Default confidence for manual additions
                    "expiration_days": None,
                    "added_timestamp": time.time(),
                    "last_updated": time.time()
                })
                new_count += 1
        elif delta < 0:
            # Remove items
            to_remove = min(abs(delta), new_count)
            for _ in range(to_remove):
                # Find and remove a non-pending item
                for i, itm in enumerate(inventory_items):
                    if itm["label"].lower() == item and not itm.get("pending", False):
                        inventory_items.pop(i)
                        new_count -= 1
        
        # Emit updated inventory
        emit_inventory()
        
        return jsonify({
            'success': True,
            'new_count': new_count
        })
        
    except Exception as e:
        logger.error(f"Error updating inventory count: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/remove_inventory_item', methods=['POST'])
def remove_inventory_item():
    try:
        data = request.json
        item = data.get('item', '').lower()  # Convert to lowercase
        
        if not item:
            return jsonify({'error': 'Item name is required'}), 400
            
        # Remove all matching items (both pending and non-pending)
        initial_length = len(inventory_items)
        
        # Remove all items with matching label, regardless of source
        inventory_items[:] = [itm for itm in inventory_items if itm["label"].lower() != item]
        
        if len(inventory_items) == initial_length:
            return jsonify({'error': 'Item not found in inventory'}), 404
            
        # Emit updated inventory
        emit_inventory(source="removed")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error removing inventory item: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_item_name', methods=['POST'])
def update_item_name():
    try:
        data = request.json
        old_name = data.get('old_name', '').lower()
        new_name = data.get('new_name', '').lower()
        
        if not old_name or not new_name:
            return jsonify({'error': 'Both old and new names are required'}), 400
            
        if old_name == new_name:
            return jsonify({'error': 'New name must be different from old name'}), 400
            
        # Update all matching items in inventory
        updated_count = 0
        for item in inventory_items:
            if item["label"].lower() == old_name:
                item["label"] = new_name
                item["last_updated"] = time.time()
                updated_count += 1
        
        if updated_count == 0:
            return jsonify({'error': 'Item not found in inventory'}), 404
            
        # Emit updated inventory
        emit_inventory()
        
        return jsonify({
            'success': True,
            'updated_count': updated_count
        })
        
    except Exception as e:
        logger.error(f"Error updating item name: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_manual_item', methods=['POST'])
def add_manual_item():
    try:
        data = request.json
        item_name = data.get('item', '').lower()  # Convert to lowercase
        count = data.get('count', 1)
        
        if not item_name:
            return jsonify({'error': 'Item name is required'}), 400
            
        # Add the item(s) to inventory
        bulk_add(item_name, count, confidence=100.0)
        
        # Emit updated inventory
        emit_inventory(source="manual_add")
        
        return jsonify({
            'success': True,
            'item': item_name,
            'count': count
        })
        
    except Exception as e:
        logger.error(f"Error adding manual item: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
if __name__=="__main__":
    socketio.run(app, host="127.0.0.1", port=5001, debug=True)
