import os
import time
import math
import logging
import base64
import re

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO

from ultralytics import YOLO
from openai import OpenAI

BOUNDARY_Y = 500            # y-coordinate of the “door” line
ZOOM_FACTOR = 1.5           # digital zoom factor
CONF_THRESHOLD = 0.5        # YOLO confidence threshold
MAX_LOST_FRAMES = 5         # drop tracks after this many missed frames
IGNORE_LABEL = "person"     # skip these detections

# Menu/prices (from script 2)
menu = {
    'banana': 5, 'black beans': 4, 'grilled chicken breast': 7,
    'milk': 2, 'orange juice': 3, 'pizza': 8,
    'potato': 3, 'salad': 5, 'spaghetti': 10, 'white rice': 5
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(
    app, cors_allowed_origins="*", async_mode='eventlet',
    logger=True, engineio_logger=True
)

# YOLO model (change path if needed)
# model = YOLO("model/food_detector_small.pt")
model = YOLO("yolov8n.pt")

# OpenAI client (ensure OPENAI_API_KEY is set in your env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_key_here"))

inventory = {}     # { label: count }
tracks = {}        # { track_id: { label, center, current_side, start_side,
                   #               has_crossed_in, has_crossed_out, lost_frames } }
tracked_screenshots = set()
last_screenshot_time = 0.0

def digital_zoom(frame, factor):
    h, w = frame.shape[:2]
    new_w, new_h = int(w/factor), int(h/factor)
    x1, y1 = (w-new_w)//2, (h-new_h)//2
    crop = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

def add_to_inventory(label):
    inventory[label] = inventory.get(label, 0) + 1

def remove_from_inventory(label):
    if label in inventory:
        inventory[label] -= 1
        if inventory[label] <= 0:
            del inventory[label]

def emit_inventory():
    socketio.emit('inventory_update', {'inventory': inventory})

def update_track_side(track_id, new_center):
    tr = tracks[track_id]
    old = tr['current_side']
    new = 'below' if new_center[1] > BOUNDARY_Y else 'above'
    tr['center'] = new_center

    # initialize start_side
    if tr['start_side'] is None:
        tr['start_side'] = new

    tr['current_side'] = new

    # crossing in: above -> below
    if old == 'above' and new == 'below' and not tr['has_crossed_in']:
        add_to_inventory(tr['label'])
        tr['has_crossed_in'] = True
        socketio.emit('movement_event', {
            'type': 'input',
            'item': tr['label']
        })
        emit_inventory()

    # crossing out: below -> above
    if old == 'below' and new == 'above' and not tr['has_crossed_out']:
        remove_from_inventory(tr['label'])
        tr['has_crossed_out'] = True
        socketio.emit('movement_event', {
            'type': 'output',
            'item': tr['label']
        })
        emit_inventory()

def encode_frame_to_base64(frame):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, buf = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buf).decode('utf-8')

def capture_detection_screenshot(frame, class_name, track_id):
    return {
        'image': encode_frame_to_base64(frame),
        'class_name': class_name,
        'track_id': track_id,
        'timestamp': time.time()
    }

def generate_frames():
    global last_screenshot_time

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        logger.error(f"Could not open camera at index {1}")
        return

    emit_interval       = 0.5
    screenshot_interval = 1.0
    last_emit_time      = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Digital zoom
        frame = digital_zoom(frame, ZOOM_FACTOR)
        h, w = frame.shape[:2]

        # 2) YOLO track/detect
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)

        current_ids    = set()
        detected_items = {}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names

            # --- CASE A: no track IDs yet ---
            if boxes.id is None:
                coords   = boxes.xyxy.cpu().tolist()
                confs    = boxes.conf.cpu().tolist()
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

            # --- CASE B: tracker IDs are present ---
            else:
                ids      = boxes.id.int().cpu().tolist()
                cls_idxs = boxes.cls.int().cpu().tolist()
                coords   = boxes.xyxy.cpu().tolist()
                confs    = boxes.conf.cpu().tolist()

                for tid, cls_idx, (x1, y1, x2, y2), conf in zip(ids, cls_idxs, coords, confs):
                    label = names[cls_idx]
                    if label == IGNORE_LABEL:
                        continue

                    cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
                    current_ids.add(tid)

                    # init or reset track
                    if tid not in tracks:
                        tracks[tid] = {
                            'label': label,
                            'center': (cx, cy),
                            'start_side': None,
                            'current_side': None,
                            'has_crossed_in': False,
                            'has_crossed_out': False,
                            'lost_frames': 0
                        }
                    else:
                        tracks[tid]['lost_frames'] = 0

                    # update crossing / inventory
                    update_track_side(tid, (cx, cy))

                    # screenshot logic (unchanged)
                    now = time.time()
                    if (tid not in tracked_screenshots
                            and now - last_screenshot_time >= screenshot_interval):
                        padding = 20
                        x1i = max(0, int(x1) - padding)
                        y1i = max(0, int(y1) - padding)
                        x2i = min(w, int(x2) + padding)
                        y2i = min(h, int(y2) + padding)
                        crop = frame[y1i:y2i, x1i:x2i]
                        if crop.size:
                            data = capture_detection_screenshot(crop, label, tid)
                            socketio.emit('detection_screenshot', data)
                            tracked_screenshots.add(tid)
                            last_screenshot_time = now

                    # draw box + label
                    x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (255, 0, 0), 2)
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

        # 4) Draw the boundary line
        cv2.line(frame, (0, BOUNDARY_Y), (w, BOUNDARY_Y), (0, 0, 255), 2)

        # 5) Emit detection counts at intervals
        now = time.time()
        if now - last_emit_time >= emit_interval:
            socketio.emit('detection_update', {
                'items': detected_items,
                'timestamp': now
            })
            last_emit_time = now

        # 6) Always yield the frame
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

# --------------------------------------------------------------------
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
