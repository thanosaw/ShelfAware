import cv2
import math
import os
import sys
import contextlib
from ultralytics import YOLO

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
BOUNDARY_Y = 500            # Horizontal line dividing "above" vs "below"
CONF_THRESHOLD = 0.4        # YOLO confidence threshold
MAX_DISTANCE = 200          # Max centroid distance to consider the same track
MAX_LOST_FRAMES = 5        # If a track is unmatched this many frames, remove it
IGNORE_LABEL = "person"     # We will completely ignore "person" detections
ZOOM_FACTOR = 2.0           # Digital zoom factor (2.0 = 2x)
FRAMES_CONFIRMATION = 3     # If you want additional consecutive-frame logic (unused here but optional)
INTERESTED_CLASSES = []     # If empty, we detect all classes (except we still skip IGNORE_LABEL)

# YOLO model (replace with your custom .pt file if needed)
model = YOLO("yolov8n.pt")

# Inventory as { label: count }
inventory = {}

# Track dictionary: { track_id: {...track_info...} }
tracks = {}
next_track_id = 0  # increment when creating a new track


# --------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------
def digital_zoom(frame, factor):
    """
    Crop the center portion of the frame by 'factor' and resize to original dimension,
    simulating a digital zoom.
    E.g., factor=2.0 => take the central 1/2 width & height, then scale it up 2x.
    """
    h, w = frame.shape[:2]
    new_w = int(w / factor)
    new_h = int(h / factor)

    # Coordinates to crop the center region
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Crop and resize back to original size
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed


def print_inventory():
    """Show the current inventory with counts."""
    print("Inventory:")
    if not inventory:
        print("  Empty")
    else:
        for item, count in inventory.items():
            print(f"  {item} x{count}")
    print()


def add_to_inventory(label):
    """Increment the count for a label in the inventory."""
    if label not in inventory:
        inventory[label] = 0
    inventory[label] += 1


def remove_from_inventory(label):
    """Decrement the count for a label; remove if it hits zero."""
    if label in inventory:
        inventory[label] -= 1
        if inventory[label] <= 0:
            del inventory[label]


def euclidean_distance(p1, p2):
    """Euclidean distance between p1=(x1,y1) and p2=(x2,y2)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def match_detections_to_tracks(detections, tracks):
    """
    Attempt to match each detection (label, center) to an existing track by label + proximity.
    Returns:
      matches: list of (det_idx, track_id)
      unmatched_detections: list of detection indices
      unmatched_tracks: list of track IDs
    """
    matches = []
    unmatched_detections = set(range(len(detections)))
    unmatched_tracks = set(tracks.keys())

    for det_idx, (label, center) in enumerate(detections):
        best_track_id = None
        best_distance = float("inf")

        # Compare to each unmatched track
        for t_id in list(unmatched_tracks):
            tr = tracks[t_id]
            # Only match if labels match
            if tr["label"] != label:
                continue
            # Check distance to track's center
            dist = euclidean_distance(center, tr["center"])
            if dist < MAX_DISTANCE and dist < best_distance:
                best_distance = dist
                best_track_id = t_id

        if best_track_id is not None:
            matches.append((det_idx, best_track_id))
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(best_track_id)

    return matches, list(unmatched_detections), list(unmatched_tracks)


def update_track_side(track, new_center):
    """
    Update the track's side (above/below) based on new_center.y.
    Check for boundary crossing events.
    """
    old_side = track["current_side"]
    new_side = "below" if new_center[1] > BOUNDARY_Y else "above"

    track["current_side"] = new_side
    track["center"] = new_center

    # If first time seeing it, set the start_side
    if track["start_side"] is None:
        track["start_side"] = new_side

    # If crossing from above->below and hasn't already "crossed in"
    if old_side == "above" and new_side == "below" and not track["has_crossed_in"]:
        add_to_inventory(track["label"])
        track["has_crossed_in"] = True
        print_inventory()

    # If crossing from below->above and hasn't already "crossed out"
    if old_side == "below" and new_side == "above" and not track["has_crossed_out"]:
        remove_from_inventory(track["label"])
        track["has_crossed_out"] = True
        print_inventory()


# --------------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------------
def main():
    global next_track_id

    # 1) Use camera index = 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam at index 1.")
        return

    print("Starting real-time detection (digital zoom). Press 'q' to quit.")
    print_inventory()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply digital zoom
        zoomed_frame = digital_zoom(frame, ZOOM_FACTOR)

        # Suppress YOLO console prints
        results = model.predict(zoomed_frame, conf=CONF_THRESHOLD, verbose=False)

        # Prepare detection list: (label, center), plus bboxes for drawing
        detection_list = []
        bboxes_for_drawing = []
        if len(results) > 0:
            detections = results[0].boxes
            names = results[0].names
            for det in detections:
                cls_id = int(det.cls[0].item())
                label = names[cls_id]
                conf = float(det.conf[0].item())

                # Ignore "person"
                if label == IGNORE_LABEL:
                    continue

                # If we have a filter, skip labels not in it
                if INTERESTED_CLASSES and label not in INTERESTED_CLASSES:
                    continue

                # Get the bounding box and center
                bbox_xyxy = det.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = bbox_xyxy
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                detection_list.append((label, (x_center, y_center)))
                bboxes_for_drawing.append((bbox_xyxy, label, conf))

        # Match detections to existing tracks
        matches, unmatched_detections, unmatched_tracks = match_detections_to_tracks(detection_list, tracks)

        # 1) Update matched tracks
        for det_idx, track_id in matches:
            label, center = detection_list[det_idx]
            track = tracks[track_id]
            track["lost_frames"] = 0
            update_track_side(track, center)

        # 2) For unmatched detections, create new tracks
        for det_idx in unmatched_detections:
            label, center = detection_list[det_idx]
            tracks[next_track_id] = {
                "label": label,
                "center": center,
                "start_side": None,
                "current_side": None,
                "lost_frames": 0,
                "has_crossed_in": False,
                "has_crossed_out": False,
            }
            update_track_side(tracks[next_track_id], center)
            next_track_id += 1

        # 3) Increment lost_frames for unmatched tracks
        for track_id in unmatched_tracks:
            tracks[track_id]["lost_frames"] += 1

        # 4) Remove tracks that are lost too long
        to_remove = [t_id for t_id, tr in tracks.items() if tr["lost_frames"] > MAX_LOST_FRAMES]
        for t_id in to_remove:
            del tracks[t_id]

        # -------------------------------------------------
        # VISUALIZE bounding boxes + boundary line + track IDs
        # -------------------------------------------------
        h, w = zoomed_frame.shape[:2]

        # Draw boundary line (still at BOUNDARY_Y)
        cv2.line(zoomed_frame, (0, BOUNDARY_Y), (w, BOUNDARY_Y), (0, 0, 255), 2)

        # Draw bounding boxes
        for (xyxy, label, conf) in bboxes_for_drawing:
            x_min, y_min, x_max, y_max = map(int, xyxy)
            cv2.rectangle(zoomed_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(
                zoomed_frame,
                text,
                (x_min, max(y_min - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # Draw track IDs (centroids)
        for t_id, tr in tracks.items():
            cx, cy = tr["center"]
            cv2.circle(zoomed_frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)
            cv2.putText(
                zoomed_frame,
                f"ID:{t_id}",
                (int(cx) + 10, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        cv2.imshow("Fridge Vision (Zoomed)", zoomed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
