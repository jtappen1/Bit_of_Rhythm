import cv2
import numpy as np
from collections import deque

class KalmanTracker:
    _id = 0
    def __init__(self, init_xy, dt=1.0):
        self.kf = cv2.KalmanFilter(6, 2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        # tune process/measurement noise
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        # initial state
        x, y = float(init_xy[0]), float(init_xy[1])
        self.kf.statePre = np.array([[x], [y], [0.], [0.], [0.], [0.]], dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0.], [0.], [0.], [0.]], dtype=np.float32)

        self.id = KalmanTracker._id
        KalmanTracker._id += 1
        self.color = tuple(int(c) for c in np.random.randint(100, 255, 3))  # bright color
        self.time_since_update = 0
        self.age = 0
        self.hits = 1

        self.max_history = 8
        self.history = deque([], maxlen=30)
        self.velocity_history = deque([], maxlen=self.max_history)
        self.acceleration_history = deque([], maxlen=self.max_history)
        self.cooldown = 0
        

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        px, py = float(pred[0]), float(pred[1])
        return (px, py)

    def update(self, meas_xy):
        meas = np.array([[np.float32(meas_xy[0])], [np.float32(meas_xy[1])]])
        corrected = self.kf.correct(meas)
        self.time_since_update = 0
        self.hits += 1
        px, py = float(corrected[0]), float(corrected[1])
        if len(self.history) == 0 or (px,py) != self.history[-1]:
            self.history.append((px,py))

        vx, vy = self.get_velocity()
        ax, ay = self.get_acceleration()
        if len(self.velocity_history) == 0 or (vx, vy) != self.velocity_history[-1]:
            self.velocity_history.append((vx, vy))
        if len(self.acceleration_history) == 0 or (ax, ay) != self.acceleration_history[-1]:
            self.acceleration_history.append((ax, ay))

        return (px, py)

    def get_state(self):
        s = self.kf.statePost.flatten()
        return float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])
    
    def get_velocity(self):
        vx, vy = self.kf.statePost[2], self.kf.statePost[3]
        return float(vx), float(vy)

    def get_position(self):
        x, y = self.kf.statePost[0], self.kf.statePost[1]
        return float(x), float(y)
    
    def get_acceleration(self):
        ax, ay = self.kf.statePost[4], self.kf.statePost[5]
        return float(ax), float(ay)
    
    def decrement_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        return (self.cooldown == 0)
    
    def set_cooldown(self):
        self.cooldown = self.max_history + 1
    
    def detect_hit(
        self,
        vel_threshold: float = 40.0,
        acc_threshold: float = 20.0,
        min_hist = 2
    ) -> tuple[bool, int]:
        # Decrement the cooldown 
        if not self.decrement_cooldown():
            return False, 0
        
        # Check if there is a long enough history to even know if a detection happened
        if len(self.velocity_history)  < min_hist and len(self.acceleration_history) < min_hist:
            return False, 0
        
        vel_hist = np.array(self.velocity_history)

        # Get the previous velocities in the y direction. 
        vy = vel_hist[:, 1]
        vy_prev = vy[:-1]
        # Get the current velocity
        vy_curr = vy[-1]  

        # Check the current velocity is going up and which ones in the previous were going down
        condition = (vy_curr < 0) & (vy_prev > 0)
        if not np.any(condition):
            return False, 0
        
        # Check if the diff is greater than the threshold
        delta_vy = np.abs(vy_prev[condition] - vy_curr)

        print(f"Trace {self.id}--{delta_vy} -- {vy}")


        # Hit Detected, set cooldown
        if np.max(delta_vy) > vel_threshold:
            if np.all(vy[-3:] < 0) and np.any(vy_prev > 0):

                signs = np.sign(vy)
                # Detect where sign changes between consecutive elements
                sign_change_mask = np.diff(signs) != 0

                # Get indices where sign changes occur (index of element *before* the change)
                sign_change_indices = np.where(sign_change_mask)[0]
                print(f"Sign Changes{sign_change_indices}")
                print(f"Diff = {self.max_history - sign_change_indices[0]}")
                      

                self.set_cooldown()
                print(f"Hit Detected! Delta:{delta_vy}")
                return True, self.max_history - sign_change_indices[0]
            else:
                print("FAILED SECOND")

        return False, 0