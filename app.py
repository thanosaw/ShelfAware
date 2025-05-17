from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO
import time
import logging
import numpy as np
import base64
import requests
import re
from openai import OpenAI
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure Socket.IO with explicit settings
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)
model = YOLO("model/food_detector_small.pt") 

menu = {
    'banana': 5,
    'black beans': 4,
    'grilled chicken breast': 7,
    'milk': 2,
    'orange juice': 3,
    'pizza': 8,
    'potato': 3,
    'salad': 5,
    'spaghetti': 10,
    'white rice': 5
}

# Global variable for the camera
camera = cv2.VideoCapture(0)

# OpenAI client (ensure OPENAI_API_KEY is set in your env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_frame_to_base64(frame):
    """Convert an OpenCV frame to base64 string with compression"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # 70% quality
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

def capture_detection_screenshot(frame, class_name, track_id):
    """Capture and encode a screenshot of the detected food item"""
    encoded_image = encode_frame_to_base64(frame)
    return {
        'image': encoded_image,
        'class_name': class_name,
        'track_id': track_id,
        'timestamp': time.time()
    }

def detect_objects(frame):
    """
    Your existing object detection function
    Should return a dictionary of detected items and their counts
    """
    # This is a placeholder - replace with your actual detection logic
    detected_items = {"apple": 1, "banana": 2}  # Example
    return detected_items

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

def generate_frames():
    camera = cv2.VideoCapture(0)
    last_emit_time = 0
    last_screenshot_time = 0  # Add screenshot timing control
    emit_interval = 0.5
    screenshot_interval = 1.0  # Take screenshots every second
    
    tracked_screenshots = set()  # Keep track of which objects we've already captured
    tracked_objects = {} # Dictionary to store tracked objects {track_id: {'class_name': name, 'center_x': x, 'last_seen': timestamp}}
    frame_width = None
    edge_threshold = 50 # Pixels from edge to consider as exit

    while True:
        success, frame = camera.read()
        if not success:
            logger.error("Failed to read frame from camera")
            break
        else:
            if frame_width is None:
                frame_width = frame.shape[1] # Get frame width once

            total_price = 0
            detected_items = {}
            # Use model.track instead of model() for object tracking
            results = model.track(frame, persist=True, conf=0.5, verbose=False) # persist=True maintains tracks across frames

            current_track_ids = set()

            # Check if tracking results exist and have IDs
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes # Get boxes from the first (and likely only) result
                track_ids = boxes.id.int().cpu().tolist() # Get track IDs
                class_indices = boxes.cls.int().cpu().tolist() # Get class indices
                xyxy_coords = boxes.xyxy.cpu().tolist() # Get coordinates
                confidences = boxes.conf.cpu().tolist() # Get confidences
                class_names = results[0].names # Get class names mapping

                for track_id, class_idx, coords, conf in zip(track_ids, class_indices, xyxy_coords, confidences):
                    current_track_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, coords)
                    class_name = class_names[class_idx]
                    confidence = float(conf)
                    center_x = (x1 + x2) / 2

                    # Update tracked object state
                    tracked_objects[track_id] = {
                        'class_name': class_name,
                        'center_x': center_x,
                        'last_seen': time.time()
                    }

                    # Compute the price based on specific detected class
                    price = menu.get(class_name, 0)
                    total_price += price

                    # also collect the class name of detected food to send to frontend
                    if class_name in detected_items:
                        detected_items[class_name] += 1
                    else:
                        detected_items[class_name] = 1

                    # draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # draw the label text
                    label = f"{class_name} ID:{track_id} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Only take screenshots for new objects we haven't seen before
                    # and respect the screenshot interval
                    if (track_id not in tracked_screenshots and 
                        time.time() - last_screenshot_time >= screenshot_interval):
                        
                        # Crop the frame to just the detection area (with some padding)
                        padding = 20
                        crop_x1 = max(0, x1 - padding)
                        crop_y1 = max(0, y1 - padding)
                        crop_x2 = min(frame.shape[1], x2 + padding)
                        crop_y2 = min(frame.shape[0], y2 + padding)
                        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        if cropped_frame.size > 0:  # Ensure we have a valid crop
                            screenshot_data = capture_detection_screenshot(cropped_frame, class_name, track_id)
                            try:
                                socketio.emit('detection_screenshot', screenshot_data)
                                tracked_screenshots.add(track_id)
                                last_screenshot_time = time.time()
                            except Exception as e:
                                logger.error(f'Failed to emit screenshot: {str(e)}')

            # Check for objects that disappeared near the edges
            previous_track_ids = set(tracked_objects.keys())
            lost_track_ids = previous_track_ids - current_track_ids

            for track_id in lost_track_ids:
                if track_id in tracked_objects: # Check if we still have info
                    track_info = tracked_objects[track_id]
                    last_center_x = track_info['center_x']
                    class_name = track_info['class_name']

                    # Check if it disappeared near left or right edge
                    if last_center_x < edge_threshold:
                        log_message = f"input ({class_name})"
                        logger.info(log_message)
                        # Emit movement event to frontend
                        try:
                            socketio.emit('movement_event', {'type': 'input', 'item': class_name, 'message': log_message})
                            logger.debug(f"Successfully emitted input event for {class_name}")
                        except Exception as e:
                            logger.error(f'Failed to emit input event: {str(e)}')
                        del tracked_objects[track_id] # Remove from tracking
                    elif last_center_x > frame_width - edge_threshold:
                        log_message = f"output ({class_name})"
                        logger.info(log_message)
                        # Emit movement event to frontend
                        try:
                            socketio.emit('movement_event', {'type': 'output', 'item': class_name, 'message': log_message})
                            logger.debug(f"Successfully emitted output event for {class_name}")
                        except Exception as e:
                            logger.error(f'Failed to emit output event: {str(e)}')
                        del tracked_objects[track_id] # Remove from tracking
                    # Optional: Handle objects disappearing elsewhere or cleanup old tracks
                    # elif time.time() - track_info['last_seen'] > 1.0: # Remove if unseen for 1 sec
                        # logger.debug(f"Track {track_id} ({class_name}) timed out.")
                        # del tracked_objects[track_id]

            # Clean up old tracked screenshots periodically
            if len(tracked_screenshots) > 50:  # Arbitrary limit
                tracked_screenshots.clear()

            # Emit updates at regular intervals to prevent overwhelming the client
            current_time = time.time()
            if current_time - last_emit_time >= emit_interval:
                logger.debug(f'Emitting detection update: {detected_items}')
                try:
                    socketio.emit('detection_update', {
                        'items': detected_items,
                        'timestamp': current_time
                    })
                    logger.debug('Successfully emitted detection update')
                except Exception as e:
                    logger.error(f'Failed to emit detection update: {str(e)}')
                last_emit_time = current_time

            # encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        logger.info("Starting image analysis request")
        image_data = request.json.get('image')
        if not image_data:
            logger.error("No image data provided in request")
            return jsonify({'error': 'No image data provided'}), 400

        logger.info("Making OpenAI API request")
        response = client.chat.completions.create(
            model="gpt-4.1",  # Update this if needed to match your API requirements
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "what is in this image? Disregard any text labels and identify it yourself. The image contains food items that are being tracked entering a fridge. Please identify the specific food item and provide a confidence level. If it is not food, say unknown."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        logger.info("Received response from OpenAI")
        logger.debug(f"Raw response: {response}")

        # Extract the response text
        response_text = response.choices[0].message.content
        logger.debug(f"Response text: {response_text}")

        # Parse the response for food item and confidence
        try:
            # More flexible parsing to handle various response formats
            food = response_text.split('(')[0].strip()
            confidence_match = re.search(r'(\d+)%?', response_text)
            confidence = int(confidence_match.group(1)) if confidence_match else 90

            result = {
                'food': food,
                'confidence': confidence,
                'raw_response': response_text  # Include raw response for debugging
            }
            
            logger.info(f"Analysis complete: {result}")
            return jsonify(result)

        except Exception as parse_error:
            logger.error(f"Error parsing OpenAI response: {parse_error}")
            return jsonify({
                'food': response_text[:50] + "...",  # Return truncated response
                'confidence': 70,  # Default confidence
                'parsing_error': str(parse_error),
                'raw_response': response_text
            })

    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
