from collections import deque
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import filedialog
from annotate import Annotator
import yaml
from trackers import TipTracker
from transcription import transcribe

COLOR_RIGHT = (0,0,255)
COLOR_LEFT = (0,255,0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def select_video_file():
    """
    Open a file dialog to select a video file.
    
    Returns:
        str: Path to the selected video file
    """
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        )
    )
    return file_path

def merge_touching_boxes(boxes, confs=None, classes=None):
    """
    Merge all touching/overlapping bounding boxes using NumPy vectorization.
    Uses a union-find algorithm to group boxes that touch or overlap, then
    merges each group into a single bounding box.
    
    Args:
        boxes: np.array of shape (N,4) with [x1, y1, x2, y2] coordinates
        confs: optional np.array of shape (N,) containing confidence scores
        classes: optional np.array of shape (N,) containing class IDs
    
    Returns:
        tuple: (merged_boxes, merged_confs, merged_classes)
            - merged_boxes: np.array of merged bounding boxes
            - merged_confs: np.array of max confidences per merged box (or None)
            - merged_classes: np.array of class IDs per merged box (or None)
    """
    if len(boxes) == 0:
        return np.empty((0,4), dtype=int), np.array([]) if confs is not None else None, np.array([]) if classes is not None else None

    boxes = np.array(boxes, dtype=int)
    N = len(boxes)
    parent = np.arange(N)  # union-find array

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    # Vectorized pairwise check for touching
    x1 = boxes[:,0][:,None]
    y1 = boxes[:,1][:,None]
    x2 = boxes[:,2][:,None]
    y2 = boxes[:,3][:,None]

    # two boxes do NOT touch if one is completely left/right/up/down of the other
    not_touch = (x2 < x1.T) | (x2.T < x1) | (y2 < y1.T) | (y2.T < y1)
    touch = ~not_touch

    # Union-find: merge indices that touch
    for i in range(N):
        touching = np.where(touch[i])[0]
        for j in touching:
            union(i,j)

    # Build groups
    groups = {}
    for i in range(N):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Merge boxes per group
    merged_boxes = []
    merged_confs = [] if confs is not None else None
    merged_classes = [] if classes is not None else None
    for inds in groups.values():
        merged_boxes.append([
            np.min(boxes[inds,0]),
            np.min(boxes[inds,1]),
            np.max(boxes[inds,2]),
            np.max(boxes[inds,3])
        ])
        if confs is not None:
            merged_confs.append(np.max(confs[inds]))
        if classes is not None:
            merged_classes.append(classes[inds[0]])

    merged_boxes = np.array(merged_boxes, dtype=int)
    if merged_confs is not None:
        merged_confs = np.array(merged_confs)
    if merged_classes is not None:
        merged_classes = np.array(merged_classes)

    return merged_boxes, merged_confs, merged_classes

def inference(config):
    """
    Main inference loop for drumstick detection and tracking.
    
    Processes video frame-by-frame to:
    1. Detect drumsticks using custom YOLO model
    2. Detect wrist positions using YOLO pose estimation
    3. Merge overlapping detections
    4. Assign left/right labels based on wrist proximity
    5. Calculate and log speed(s) to CSV
    6. Display annotated video with frame-by-frame navigation
    
    Controls:
        - 'q': Quit
        - Any other key: Next frame
    """
    
    model = YOLO("PATH")

    video_path = select_video_file()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    tip_tracker = TipTracker(dist_thresh=config['dist_thresh'], max_age=config['max_age'])
    annotator = Annotator()
    timestamps = []
    
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    # For each frame, store [left_stick_y, right_stick_y]
    distance_from_top = np.zeros(shape=(total_frame_count, 2))

    previous_frames = deque(maxlen=8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, stream=True, verbose=False)

        annotator.set_frame(frame)
        
        all_boxes, all_confs, all_classes = [], [], []
        
        # Get all boxes, confidences, and classes for each detected object
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            # Filter by confidence threshold
            mask = confs >= config["min_confidence"]
            all_boxes.append(boxes[mask])
            all_confs.append(confs[mask])
            all_classes.append(classes[mask])

        # Combine all detections
        all_boxes = np.vstack(all_boxes)
        all_confs = np.hstack(all_confs)
        all_classes = np.hstack(all_classes)

        # Merge duplicate boxes that are touching
        merged_boxes, merged_confs, merged_classes = merge_touching_boxes(all_boxes, all_confs, all_classes)
        
        detections = []
        for idx, (box, conf, cls) in enumerate(zip(merged_boxes, merged_confs, merged_classes)):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            
            # Calculate current center coordinates
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2 )
            detections.append((x_center, y_center))

            # Drawing a dot where the computed center is.
            cv2.circle(frame, (x_center, y_center), 16, (0, 0, 255), -1)

            # Distance save
            distance_from_top[frame_count][cls_id] = y_center

        tip_tracker.update(detections)
        hits = tip_tracker.detect_hits()

        for _, idx in hits:
            timestamps.append(frame_count - idx/fps)
            print(idx)
            cv2.imshow("Hit Detected", previous_frames[7-idx])

        # Annotate KF information on screen
        annotator.annotate_trackers(tip_tracker.trackers, hits)
        # Annotate boxes and confidences on screen
        annotator.annotate_bounding_boxes(merged_boxes, merged_confs)

        annotator.annotate_time(frame_count/fps)

        frame = annotator.get_frame()
        
        # --- 3. Display and Exit (Unchanged) ---
        cv2.imshow("YOLOv8 Live", frame)
        frame_count += 1
        previous_frames.append(frame)

        key = cv2.waitKey(0)
        if  key == ord('q'):
            break
        elif  key == ord('n'):
            continue


    cap.release()
    cv2.destroyAllWindows()

    # Run transcription
    transcribe(data=distance_from_top)

if __name__ == "__main__":

    with open('PATH', 'r') as file:
        config = yaml.safe_load(file)

    inference(config)