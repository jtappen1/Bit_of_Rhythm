import math
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
from tkinter import filedialog
from kalman_filter import KalmanTracker
from enum import Enum
from collections import deque

COLOR_RIGHT = (0,0,255)
COLOR_LEFT = (0,255,0)

class StickTip(Enum):
    LEFT = 0
    RIGHT = 1

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

class StickTracker:
    """
    Tracks drumstick positions and assigns left/right labels based on wrist proximity.
    Uses YOLO pose estimation to detect wrists and calculate distances to drumstick centers.
    """
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
        self.left_tip = False
        self.right_tip = False
        self.distances = {}
    
    def reset_state(self):
        """Reset tracking state for a new frame."""
        self.left_tip = False
        self.right_tip = False
        self.distances = {}

    def get_wrist_coords(self, frame):
        """
        Detect and annotate wrist coordinates using pose estimation.
        
        Args:
            frame: Video frame to process (numpy array)
        
        Returns:
            tuple: (right_wrist_coords, left_wrist_coords, annotated_frame)
                - right_wrist_coords: tuple (x, y) or None if not detected
                - left_wrist_coords: tuple (x, y) or None if not detected
                - annotated_frame: Frame with wrist circles drawn
        """
        self.reset_state()

        poses = self.model(frame, verbose=False)

        left_wrist_coords: tuple | None  = None
        right_wrist_coords: tuple | None = None

        for pose in poses:
            if pose.keypoints:
                keypoints = pose.keypoints.xy.cpu().numpy()
                confs = pose.keypoints.conf.cpu().numpy()
                
                lx, ly = map(int, keypoints[0][9]) # Left Wrist
                rx, ry = map(int, keypoints[0][10]) # Right Wrist
                l_conf = confs[0][9]
                r_conf = confs[0][10]
                
                if l_conf > 0.5:
                    cv2.circle(frame, center=(lx, ly), radius=20, color= (0, 0 , 255), thickness=-1)
                    left_wrist_coords = (lx, ly)
                    
                if r_conf > 0.5:
                    cv2.circle(frame, center=(rx, ry), radius=20, color= (0, 255 , 0), thickness=-1)
                    right_wrist_coords = (rx, ry)

        return right_wrist_coords, left_wrist_coords, frame
    
    def determine_tip_distances(self, left_wrist_coords, right_wrist_coords, center_coords, idx) -> None:
        """
        Calculate Euclidean distances from drumstick center to each wrist.
        
        Args:
            left_wrist_coords: tuple (x, y) or None for left wrist position
            right_wrist_coords: tuple (x, y) or None for right wrist position
            center_coords: np.array [x, y] for drumstick center
        """
        left_dist = np.linalg.norm(left_wrist_coords - center_coords) if left_wrist_coords is not None else float('inf')
        right_dist = np.linalg.norm(right_wrist_coords - center_coords) if right_wrist_coords is not None else float('inf')
        self.distances[idx] = (left_dist, right_dist, center_coords[0], center_coords[1])
    
    def get_min_distances(self):
        """
        Find which drumstick (by index) is closest to each wrist.
        
        Returns:
            tuple: (min_left_key, min_right_key) - indices of drumsticks closest to 
                   left and right wrists respectively
        """
        min_left_key = min(self.distances, key=lambda k: self.distances[k][0])
        min_right_key = min(self.distances, key=lambda k: self.distances[k][1])
        return min_left_key, min_right_key
    
    def get_distances(self, key):
        return self.distances[key]
    

class VelocityTracker:
    # Note: This needs some work, I want to end tracking after X frames or no  update, etc.

    def __init__(self, hit_cooldown = 3):
        self.left_tracker = None
        self.right_tracker = None
        self.history = {
            StickTip.LEFT: deque(maxlen=5),
            StickTip.RIGHT: deque(maxlen=5)
        }
        self.last_hit_time = {StickTip.LEFT: 0, StickTip.RIGHT: 0}
        self.hit_cooldown = hit_cooldown
        self.cooldown = {
            StickTip.LEFT: 0,
            StickTip.RIGHT: 0 
        }

    def start_tracker(self, tip: StickTip, xy: tuple[float, float]):
        if tip == StickTip.LEFT and not self.left_tracker:
            self.left_tracker = KalmanTracker(xy)
        if tip == StickTip.RIGHT and not self.right_tracker:
            self.right_tracker = KalmanTracker(xy)
    
    def update(self, tip: StickTip, xy: tuple[float, float]):
        self.start_tracker(tip=tip, xy=xy)
        tracker = self.left_tracker if tip == StickTip.LEFT else self.right_tracker
        tracker.update(xy)
        x, y = tracker.get_position()
        self.history[tip].append((x, y))

    def detect_hit(self, frame, tip: StickTip):
        history = list(self.history[tip])
        if len(history) < 4:
            return False

        xy2, xy1, xy0 = history[-3], history[-2], history[-1]
        
        dx_v1 = xy1[0] - xy2[0]
        dy_v1 = xy1[1] - xy2[1]
        angle1 = math.atan2(dy_v1, dx_v1)  
        angle_deg1 = math.degrees(angle1)

        dx_v2 = xy0[0] - xy1[0]
        dy_v2 = xy0[1] - xy1[1]
        angle2 = math.atan2(dy_v2, dx_v2)  
        angle_deg2 = math.degrees(angle2)

        # Blue Arrow x2 -> x1
        cv2.arrowedLine(frame, (int(xy2[0]), int(xy2[1])), (int(xy1[0]), int(xy1[1])), (255,0,0), 2, tipLength=0.3)

        # White Current Arrow x1 -> x_curr
        cv2.arrowedLine(frame, (int(xy1[0]), int(xy1[1])), (int(xy0[0]), int(xy0[1])), (255,255,255), 2, tipLength=0.3)

        if  (0 <= angle_deg1 < 180) and (-180 <= angle_deg2 <= -1):
            if self.cooldown[tip] == 0:
                self.cooldown[tip] = self.hit_cooldown + 1
                return True
            
        if self.cooldown[tip] != 0:
            self.cooldown[tip] -= 1

        return False
    
    def predict(self):
        if self.left_tracker:
            self.left_tracker.predict()
        if self.right_tracker:
            self.right_tracker.predict()
    
    def annotate_direction(self, frame, tip: StickTip, color=(255, 255, 255)):
        tracker = self.left_tracker if tip == StickTip.LEFT else self.right_tracker
        x, y = tracker.get_position()
        vx, vy = tracker.get_velocity()
        end_x = int(x + vx * 5)
        end_y = int(y + vy * 5)
        cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), color, 2, tipLength=0.3)
         
        if self.detect_hit(frame, tip):
            cv2.circle(frame, (int(x), int(y)), 20, (0, 255, 0), 4)
            cv2.putText(frame, "HIT!", (int(x) - 20, int(y) - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        return frame
    

def annotate_bounding_box(frame, bounding_box, class_name, conf, velocity, color):
    """
    Draw a bounding box with label on the frame.
    
    Args:
        frame: Video frame to annotate
        box: list of bounding box coordinates [x1, y1, x2, y2]
        class_name: Label text (e.g., "left", "right")
        conf: Confidence score
        velocity: Velocity in pixels per second (CURRENTLY ZERO)
        color: BGR color tuple for the box and text
    
    Returns:
        Annotated frame
    """
    x1, y1, x2, y2 = map(int, bounding_box)
    label_text = f"{class_name} Conf:{conf:.2f} Vel:{velocity:.2f} p/s"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def inference():
    """
    Main inference loop for drumstick detection and tracking.
    
    Processes video frame-by-frame to:
    1. Detect drumsticks using custom YOLO model
    2. Detect wrist positions using YOLO pose estimation
    3. Merge overlapping detections
    4. Assign left/right labels based on wrist proximity
    5. Calculate and log velocities to CSV
    6. Display annotated video with frame-by-frame navigation
    
    Controls:
        - 'q': Quit
        - Any other key: Next frame
    """
    model = YOLO("/Users/jtappen/Projects/Bit_of_Rhythm/deep-learning/weights/test1/best.pt")
    stick_tracker = StickTracker()
    velocity_tracker = VelocityTracker()

    video_path = select_video_file()
    cap = cv2.VideoCapture(video_path)

    # Tracking setup
    # Stores {class_id: {'center': (x, y), 'time': timestamp}}
    last_positions = {}
    
    # CSV file setup
    csv_filename = "box_velocities.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(['Timestamp', 'Class_ID', 'Label', 'Velocity (pixels/sec)'])

    print(f"Tracking velocities and saving to {csv_filename}...")
    
    frame_count = 0
    start_time = time.time() # To keep track of the overall runtime

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()

        results = model(frame, stream=True, verbose=False)
        left_wrist_coords, right_wrist_coords, frame = stick_tracker.get_wrist_coords(frame=frame)
        
        all_boxes, all_confs, all_classes = [], [], []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            # Filter by confidence threshold
            mask = confs >= 0.35
            all_boxes.append(boxes[mask])
            all_confs.append(confs[mask])
            all_classes.append(classes[mask])

        # Combine all detections
        all_boxes = np.vstack(all_boxes)
        all_confs = np.hstack(all_confs)
        all_classes = np.hstack(all_classes)

        merged_boxes, merged_confs, merged_classes = merge_touching_boxes(all_boxes, all_confs, all_classes)
          
        for idx, (box, conf, cls) in enumerate(zip(merged_boxes, merged_confs, merged_classes)):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            label = model.names[cls_id]

            # Calculate current center coordinates
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2 )
            center_coords = np.array([x_center, y_center])

            # Drawing a dot where the computed center is.
            cv2.circle(frame, (x_center, y_center), 8, (0, 0, 255), -1)

            # Calculate distances from center to left and right wrist.
            # Save in Stick Tracker
            stick_tracker.determine_tip_distances(
                left_wrist_coords=left_wrist_coords, 
                right_wrist_coords=right_wrist_coords, 
                center_coords=center_coords,
                idx=idx
            )
            
            # NOTE: This will no longer work.
            velocity = 0.0
            
            if cls_id in last_positions:
                # Retrieve previous data
                prev_center = last_positions[cls_id]['center']
                prev_time = last_positions[cls_id]['time']
                
                time_diff = current_time - prev_time
                
                if time_diff > 0:
                    # Calculate Euclidean distance between centers (in pixels)
                    distance = np.linalg.norm(center_coords - prev_center)
                    # Velocity in pixels per second
                    velocity = distance / time_diff
                    
                    # Write to CSV
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f"{current_time - start_time:.4f}", cls_id, label, f"{velocity:.2f}"])
                        
            # Update last position for the current class ID
            last_positions[cls_id] = {'center': center_coords, 'time': current_time}

       # Get box indices for left and right drumsticks. This will correspond to the merged boxes, 
       # letting us know which boxes are closest to which wrist.
        left_index, right_index = stick_tracker.get_min_distances()

        # If same box is closest to both wrists, assign to the closer one
        if left_index == right_index:
            left_dist, right_dist, center_x, center_y = stick_tracker.get_distances(left_index)
            tip = StickTip.LEFT if left_dist <= right_dist else StickTip.RIGHT
            color = COLOR_LEFT if left_dist <= right_dist else COLOR_RIGHT
            frame = annotate_bounding_box(frame, merged_boxes[left_index], StickTip(tip).name, 
                                        merged_confs[left_index], 0.0, color)
            # Update respective Kalman Filter
            velocity_tracker.update(tip=tip, xy=(center_x, center_y))
            frame = velocity_tracker.annotate_direction(frame, tip, color)

        else:
            # Annotate both drumsticks
            frame = annotate_bounding_box(frame, merged_boxes[left_index], StickTip.LEFT.name, 
                                        merged_confs[left_index], 0.0, COLOR_LEFT)
            frame = annotate_bounding_box(frame, merged_boxes[right_index], StickTip.RIGHT.name, 
                                        merged_confs[right_index], 0.0, COLOR_RIGHT)
            
            # Update KF for both left and right drumsticks
            left_dist, right_dist, center_x, center_y = stick_tracker.get_distances(left_index)
            velocity_tracker.update(tip=StickTip.LEFT, xy=(center_x, center_y))
            frame = velocity_tracker.annotate_direction(frame, StickTip.LEFT, COLOR_LEFT)

            left_dist, right_dist, center_x, center_y = stick_tracker.get_distances(right_index)
            velocity_tracker.update(tip=StickTip.RIGHT, xy=(center_x, center_y))
            frame = velocity_tracker.annotate_direction(frame, StickTip.RIGHT, COLOR_RIGHT)
        
        velocity_tracker.predict()
        
        # --- 3. Display and Exit (Unchanged) ---
        cv2.imshow("YOLOv8 Live", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()