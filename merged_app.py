import os
import time
import math
import logging
import base64
import re
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO
from openai import OpenAI

# Constants for computer vision
BOUNDARY_Y = 500            # y-coordinate of the "door" line
ZOOM_FACTOR = 1.0          # digital zoom factor
CONF_THRESHOLD = 0.5       # YOLO confidence threshold
MAX_LOST_FRAMES = 5        # drop tracks after this many missed frames
IGNORE_LABEL = "person"    # skip these detections
MIN_TRACK_DISTANCE = 50    # minimum distance to consider a new track
MAX_TRACK_DISTANCE = 200   # maximum distance to match existing tracks
TRACK_HISTORY = 10         # number of previous positions to track
EMIT_INTERVAL = 0.5        # seconds between detection updates
SCREENSHOT_INTERVAL = 1.0  # seconds between screenshots

# Menu/prices (from script 2)
menu = {
    'banana': 5, 'black beans': 4, 'grilled chicken breast': 7,
    'milk': 2, 'orange juice': 3, 'pizza': 8,
    'potato': 3, 'salad': 5, 'spaghetti': 10, 'white rice': 5
}

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(
    app, cors_allowed_origins="*", async_mode='eventlet',
    logger=True, engineio_logger=True
)

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# OpenAI client (ensure OPENAI_API_KEY is set in your env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_api_key_here"))

# Global state
inventory = {}     # { label: {count: int, images: list, last_updated: float} }
tracks = {}        # { track_id: { label, center, current_side, start_side,
                   #               has_crossed_in, has_crossed_out, lost_frames,
                   #               history: [(x,y,timestamp)], box } }
tracked_screenshots = set()
last_screenshot_time = 0.0

def digital_zoom(frame, factor):
    """Apply digital zoom with improved edge handling."""
    h, w = frame.shape[:2]
    new_w, new_h = int(w/factor), int(h/factor)
    x1, y1 = (w-new_w)//2, (h-new_h)//2
    crop = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

def calculate_velocity(track_history):
    """Calculate object velocity from track history."""
    if len(track_history) < 2:
        return 0
    dx = track_history[-1][0] - track_history[0][0]
    dy = track_history[-1][1] - track_history[0][1]
    dt = track_history[-1][2] - track_history[0][2]
    if dt == 0:
        return 0
    return math.sqrt(dx*dx + dy*dy) / dt

def predict_next_position(track_history):
    """Predict next position based on velocity."""
    if len(track_history) < 2:
        return None
    v = calculate_velocity(track_history)
    if v == 0:
        return None
    last_pos = track_history[-1]
    dt = time.time() - last_pos[2]
    dx = v * dt * (track_history[-1][0] - track_history[-2][0]) / math.sqrt(
        (track_history[-1][0] - track_history[-2][0])**2 + 
        (track_history[-1][1] - track_history[-2][1])**2
    )
    dy = v * dt * (track_history[-1][1] - track_history[-2][1]) / math.sqrt(
        (track_history[-1][0] - track_history[-2][0])**2 + 
        (track_history[-1][1] - track_history[-2][1])**2
    )
    return (last_pos[0] + dx, last_pos[1] + dy)

def add_to_inventory(label, image=None):
    """Add item to inventory with image and timestamp."""
    if label not in inventory:
        inventory[label] = {'count': 0, 'images': [], 'last_updated': time.time()}
    inventory[label]['count'] += 1
    if image is not None:
        inventory[label]['images'].append(image)
    inventory[label]['last_updated'] = time.time()
    logger.info(f"Added {label} to inventory. Count: {inventory[label]['count']}")

def remove_from_inventory(label):
    """Remove item from inventory with validation."""
    if label in inventory:
        if inventory[label]['count'] > 0:
            inventory[label]['count'] -= 1
            inventory[label]['last_updated'] = time.time()
            logger.info(f"Removed {label} from inventory. Count: {inventory[label]['count']}")
            if inventory[label]['count'] <= 0:
                del inventory[label]
                logger.info(f"Removed {label} from inventory (empty)")

def emit_inventory():
    """Emit inventory update with additional metadata."""
    socketio.emit('inventory_update', {
        'inventory': inventory,
        'timestamp': time.time()
    })

def update_track_side(track_id, new_center, frame=None):
    """Update track position with improved crossing detection."""
    tr = tracks[track_id]
    old = tr['current_side']
    new = 'below' if new_center[1] > BOUNDARY_Y else 'above'
    tr['center'] = new_center

    # Update track history
    tr['history'].append((new_center[0], new_center[1], time.time()))
    if len(tr['history']) > TRACK_HISTORY:
        tr['history'].pop(0)

    # Calculate velocity and predict next position
    velocity = calculate_velocity(tr['history'])
    predicted_pos = predict_next_position(tr['history'])

    # Initialize start_side
    if tr['start_side'] is None:
        tr['start_side'] = new

    tr['current_side'] = new

    # Enhanced crossing detection with velocity check
    if old == 'above' and new == 'below' and not tr['has_crossed_in']:
        # Verify crossing with velocity and prediction
        if velocity > 0 and predicted_pos and predicted_pos[1] > BOUNDARY_Y:
            if frame is not None:
                padding = 20
                x1, y1, x2, y2 = map(int, tr['box'])
                x1i = max(0, x1 - padding)
                y1i = max(0, y1 - padding)
                x2i = min(frame.shape[1], x2 + padding)
                y2i = min(frame.shape[0], y2 + padding)
                crop = frame[y1i:y2i, x1i:x2i]
                if crop.size:
                    image_data = encode_frame_to_base64(crop)
                    add_to_inventory(tr['label'], image_data)
                else:
                    add_to_inventory(tr['label'])
            else:
                add_to_inventory(tr['label'])
            tr['has_crossed_in'] = True
            socketio.emit('movement_event', {
                'type': 'input',
                'item': tr['label'],
                'velocity': velocity,
                'timestamp': time.time()
            })
            emit_inventory()

    if old == 'below' and new == 'above' and not tr['has_crossed_out']:
        # Verify crossing with velocity and prediction
        if velocity > 0 and predicted_pos and predicted_pos[1] < BOUNDARY_Y:
            remove_from_inventory(tr['label'])
            tr['has_crossed_out'] = True
            socketio.emit('movement_event', {
                'type': 'output',
                'item': tr['label'],
                'velocity': velocity,
                'timestamp': time.time()
            })
            emit_inventory()

def encode_frame_to_base64(frame):
    """Encode frame with improved compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, buf = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buf).decode('utf-8')

def capture_detection_screenshot(frame, class_name, track_id):
    """Capture screenshot with enhanced metadata."""
    return {
        'image': encode_frame_to_base64(frame),
        'class_name': class_name,
        'track_id': track_id,
        'timestamp': time.time(),
        'confidence': tracks[track_id].get('confidence', 0.0)
    }

def generate_frames():
    """Generate video frames with enhanced tracking."""
    global last_screenshot_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera")
        return

    last_emit_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Digital zoom
        frame = digital_zoom(frame, ZOOM_FACTOR)
        h, w = frame.shape[:2]

        # 2) YOLO track/detect
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)

        current_ids = set()
        detected_items = {}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names

            if boxes.id is None:
                # Initial detection
                coords = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                cls_idxs = boxes.cls.int().cpu().tolist()

                for (x1, y1, x2, y2), conf, cls_idx in zip(coords, confs, cls_idxs):
                    label = names[cls_idx]
                    if label == IGNORE_LABEL:
                        continue

                    x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (255, 0, 0), 2)
                    text = f"{label} ({conf:.2f})"
                    cv2.putText(frame, text, (x1_i, y1_i - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    detected_items[label] = detected_items.get(label, 0) + 1
            else:
                # Tracking
                ids = boxes.id.int().cpu().tolist()
                cls_idxs = boxes.cls.int().cpu().tolist()
                coords = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()

                for tid, cls_idx, (x1, y1, x2, y2), conf in zip(ids, cls_idxs, coords, confs):
                    label = names[cls_idx]
                    if label == IGNORE_LABEL:
                        continue

                    cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
                    current_ids.add(tid)

                    # Initialize or update track
                    if tid not in tracks:
                        tracks[tid] = {
                            'label': label,
                            'center': (cx, cy),
                            'start_side': None,
                            'current_side': None,
                            'has_crossed_in': False,
                            'has_crossed_out': False,
                            'lost_frames': 0,
                            'box': (x1, y1, x2, y2),
                            'history': [(cx, cy, time.time())],
                            'confidence': float(conf)
                        }
                    else:
                        tracks[tid]['lost_frames'] = 0
                        tracks[tid]['box'] = (x1, y1, x2, y2)
                        tracks[tid]['confidence'] = float(conf)

                    # Update crossing / inventory
                    update_track_side(tid, (cx, cy), frame)

                    # Draw enhanced visualization
                    x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (255, 0, 0), 2)
                    
                    # Draw track history
                    if len(tracks[tid]['history']) > 1:
                        for i in range(1, len(tracks[tid]['history'])):
                            prev = tracks[tid]['history'][i-1]
                            curr = tracks[tid]['history'][i]
                            cv2.line(frame, 
                                   (int(prev[0]), int(prev[1])),
                                   (int(curr[0]), int(curr[1])),
                                   (0, 255, 0), 1)
                    
                    # Draw velocity vector
                    velocity = calculate_velocity(tracks[tid]['history'])
                    if velocity > 0:
                        angle = math.atan2(cy - tracks[tid]['history'][-2][1],
                                         cx - tracks[tid]['history'][-2][0])
                        length = min(50, velocity * 10)
                        end_x = int(cx + length * math.cos(angle))
                        end_y = int(cy + length * math.sin(angle))
                        cv2.arrowedLine(frame, (int(cx), int(cy)),
                                      (end_x, end_y), (0, 255, 255), 2)

                    text = f"{label} ID:{tid} ({conf:.2f})"
                    cv2.putText(frame, text, (x1_i, y1_i - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    detected_items[label] = detected_items.get(label, 0) + 1

        # 3) Cleanup lost tracks
        for tid in list(tracks):
            if tid not in current_ids:
                tracks[tid]['lost_frames'] += 1
                if tracks[tid]['lost_frames'] > MAX_LOST_FRAMES:
                    del tracks[tid]

        # 4) Draw boundary line
        cv2.line(frame, (0, BOUNDARY_Y), (w, BOUNDARY_Y), (0, 0, 255), 2)

        # 5) Emit detection counts at intervals
        now = time.time()
        if now - last_emit_time >= EMIT_INTERVAL:
            socketio.emit('detection_update', {
                'items': detected_items,
                'timestamp': now
            })
            last_emit_time = now

        # 6) Yield the frame
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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

# --------------------------------------------------------------------
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
