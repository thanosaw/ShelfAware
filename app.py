from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
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

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

def generate_frames():
    camera = cv2.VideoCapture(0)  # Remove cv2.CAP_DSHOW for macOS compatibility
    last_emit_time = 0
    emit_interval = 0.5  # Emit updates every 0.5 seconds

    while True:
        success, frame = camera.read()
        if not success:
            logger.error("Failed to read frame from camera")
            break
        else:
            total_price = 0
            detected_items = {}
            results = model(frame, conf=0.5, stream=True)

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_idx = int(box.cls[0])  # class index
                        class_name = result.names[class_idx]  # class name from index
                        confidence = float(box.conf[0])  # confidence score

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
                        label = "{} ({:.2f})".format(class_name, confidence)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
