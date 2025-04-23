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

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from openai import OpenAI

BOUNDARY_Y = 500
ZOOM_FACTOR = 1.5
CONF_THRESHOLD = 0.5
MAX_LOST_FRAMES = 5
IGNORE_LABEL = "person"

menu = {
    'banana': 5, 'black beans': 4, 'grilled chicken breast': 7,
    'milk': 2, 'orange juice': 3, 'pizza': 8,
    'potato': 3, 'salad': 5, 'spaghetti': 10, 'white rice': 5
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_THRESHOLD
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
predictor = DefaultPredictor(cfg)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_key_here"))

inventory = {}
tracks = {}
tracked_screenshots = set()
last_screenshot_time = 0.0

def digital_zoom(frame, factor):
    h, w = frame.shape[:2]
    new_w, new_h = int(w/factor), int(h/factor)
    x1, y1 = (w-new_w)//2, (h-new_h)//2
    crop = frame[y1:y1+new_h, x1:x1+new_h]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

def encode_frame_to_base64(frame):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, buf = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buf).decode('utf-8')

def generate_frames():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        logger.error("Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = digital_zoom(frame, ZOOM_FACTOR)
        outputs = predictor(frame)

        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        vis_frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        frame = vis_frame.get_image()[:, :, ::-1]

        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
                    { "type": "text", "text": (
                        "what is in this image? Disregard any text labels "
                        "and identify it yourself. The image contains food "
                        "items that are being tracked entering a fridge. "
                        "Please identify the specific food item and provide "
                        "a confidence level. If it is not food, say unknown."
                    )},
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"} }
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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)