import cv2
import numpy as np

class KalmanTracker:
    _id = 0
    def __init__(self, init_xy, dt=1.0):
        # 4 state: x, y, vx, vy | 2 measurements: x, y
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],
                                             [0,1,0,dt],
                                             [0,0,1,0],
                                             [0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],
                                              [0,1,0,0]], dtype=np.float32)
        # tune process/measurement noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        # initial state
        x, y = float(init_xy[0]), float(init_xy[1])
        self.kf.statePre = np.array([[x], [y], [0.], [0.]], dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0.], [0.]], dtype=np.float32)

        self.id = KalmanTracker._id
        KalmanTracker._id += 1
        self.color = tuple(int(c) for c in np.random.randint(100, 255, 3))  # bright color
        self.time_since_update = 0
        self.age = 0
        self.hits = 1
        self.history = []  # store last positions

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        px, py = float(pred[0]), float(pred[1])
        self.history.append((px, py))
        if len(self.history) > 30:
            self.history.pop(0)
        return (px, py)

    def update(self, meas_xy):
        meas = np.array([[np.float32(meas_xy[0])], [np.float32(meas_xy[1])]])
        corrected = self.kf.correct(meas)
        self.time_since_update = 0
        self.hits += 1
        px, py = float(corrected[0]), float(corrected[1])
        if len(self.history) == 0 or (px,py) != self.history[-1]:
            self.history.append((px,py))
        return (px, py)

    def get_state(self):
        # returns x, y, vx, vy
        s = self.kf.statePost.flatten()
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])
    
    def get_velocity(self):
        vx, vy = self.kf.statePost[2], self.kf.statePost[3]
        return float(vx), float(vy)

    def get_position(self):
        x, y = self.kf.statePost[0], self.kf.statePost[1]
        return float(x), float(y)


# -------------------- MultiTracker manager --------------------
class MultiTracker:
    def __init__(self, dist_thresh=60.0, max_age=6):
        self.trackers = []
        self.dist_thresh = dist_thresh
        self.max_age = max_age

    def predict(self):
        preds = []
        for t in self.trackers:
            preds.append(t.predict())
        return preds

    def update(self, detections):
        """
        detections: list of (x, y) tuples (raw measurements)
        """
        dets = [np.array(d) for d in detections]
        N = len(self.trackers)
        M = len(dets)

        # if no existing trackers -> spawn trackers for all detections
        if N == 0:
            for d in dets:
                self.trackers.append(KalmanTracker(d))
            return

        # build cost matrix between tracker predictions and detections
        preds = [np.array([t.kf.statePre[0,0], t.kf.statePre[1,0]]) for t in self.trackers]
        cost = np.full((N, M), np.inf, dtype=float)
        for i, p in enumerate(preds):
            for j, d in enumerate(dets):
                cost[i, j] = np.linalg.norm(p - d)

        # greedy matching: pick smallest cost pair while < threshold
        matched_tr = set()
        matched_det = set()
        cost_copy = cost.copy()
        while True:
            idx = np.unravel_index(np.argmin(cost_copy), cost_copy.shape)
            i, j = idx
            if cost_copy[i, j] == np.inf or cost_copy[i, j] > self.dist_thresh:
                break
            # assign
            self.trackers[i].update(dets[j])
            matched_tr.add(i)
            matched_det.add(j)
            # invalidate row/col
            cost_copy[i, :] = np.inf
            cost_copy[:, j] = np.inf

        # unmatched detections -> create new trackers
        for j in range(M):
            if j not in matched_det:
                self.trackers.append(KalmanTracker(dets[j]))

        # increment time_since_update for unmatched trackers and remove old ones
        to_remove = []
        for i, t in enumerate(self.trackers):
            if i not in matched_tr:
                t.time_since_update += 1
            if t.time_since_update > self.max_age:
                to_remove.append(t)
        for t in to_remove:
            self.trackers.remove(t)

    def get_tracks(self):
        out = []
        for t in self.trackers:
            x, y, vx, vy = t.get_state()
            out.append({'id': t.id, 'xy': (x,y), 'vxvy': (vx,vy), 'color': t.color, 'history': t.history})
        return out