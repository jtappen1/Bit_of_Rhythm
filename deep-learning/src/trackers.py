from enum import Enum
import math
import cv2
import numpy as np
from ultralytics import YOLO
from kalman_filter import KalmanTracker
from collections import deque

class StickTip(Enum):
    LEFT = 0
    RIGHT = 1

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
        if self.distances:
            min_left_key = min(self.distances, key=lambda k: self.distances[k][0])
            min_right_key = min(self.distances, key=lambda k: self.distances[k][1])
            return min_left_key, min_right_key
        else:
            return None, None
    
    def get_distances(self, key):
        return self.distances[key]
    

class VelocityTracker:
    # Note: This needs some work, I want to end tracking after X frames or no  update, etc.
    def __init__(self, hit_cooldown = 3, fps = 30):
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
        self.fps = fps
        self.recording = False
        self.starting_idx = 0
        self.hits = []

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
    
    def annotate_direction(self, frame, frame_idx,  tip: StickTip, color=(255, 255, 255) ):
        tracker = self.left_tracker if tip == StickTip.LEFT else self.right_tracker
        x, y = tracker.get_position()
        vx, vy = tracker.get_velocity()
        end_x = int(x + vx * 5)
        end_y = int(y + vy * 5)
        cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), color, 2, tipLength=0.3)

        if self.recording:
            time_s = (frame_idx - self.starting_idx) / self.fps
            cv2.putText(frame,  f"{time_s}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
         
        if self.detect_hit(frame, tip):
            cv2.circle(frame, (int(x), int(y)), 20, (0, 255, 0), 4)
            cv2.putText(frame, "HIT!", (int(x) - 20, int(y) - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            if self.recording == False:
                self.starting_idx = frame_idx
                self.recording = True

            time_s = (frame_idx - self.starting_idx) / self.fps
            self.hits.append(time_s)

        return frame
    
    def get_hit_timestamps(self):
        return self.hits