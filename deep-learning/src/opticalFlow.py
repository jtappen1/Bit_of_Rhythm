import cv2
import numpy as np


class OpticalFlowTracker:
    """
    Tracks object speed using dense optical flow within a bounding box region.
    Calculates the magnitude of motion vectors to determine speed in pixels per frame.
    """
    
    def __init__(self, bbox, frame_gray):
        """
        Initialize the optical flow tracker.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            frame_gray: First grayscale frame for initialization
        """
        self.prev_gray = frame_gray.copy()
        self.bbox = bbox
        self.speed = 0.0
        
    def update(self, bbox, frame_gray):
        """
        Update the tracker with a new frame and calculate speed.
        
        Args:
            bbox: Current bounding box coordinates [x1, y1, x2, y2]
            frame_gray: Current grayscale frame
            
        Returns:
            float: Speed in pixels per frame (magnitude of motion)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bounding box is within frame boundaries
        h, w = frame_gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Check if bounding box is valid
        if x2 <= x1 or y2 <= y1:
            self.prev_gray = frame_gray.copy()
            return 0.0
        
        # Extract ROI from previous and current frames
        prev_roi = self.prev_gray[y1:y2, x1:x2]
        curr_roi = frame_gray[y1:y2, x1:x2]
        
        # Check if ROI is valid
        if prev_roi.size == 0 or curr_roi.size == 0:
            self.prev_gray = frame_gray.copy()
            return 0.0
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, 
            curr_roi,
            None,
            pyr_scale=0.5,      # Image scale (<1) to build pyramids
            levels=3,           # Number of pyramid layers
            winsize=15,         # Averaging window size
            iterations=3,       # Number of iterations at each pyramid level
            poly_n=5,           # Size of pixel neighborhood
            poly_sigma=1.2,     # Standard deviation for Gaussian
            flags=0
        )
        
        # Calculate magnitude of flow vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate average speed across the ROI
        self.speed = np.mean(mag)
        
        # Update previous frame
        self.prev_gray = frame_gray.copy()
        
        return self.speed
    
    def get_speed(self):
        """
        Get the last calculated speed.
        
        Returns:
            float: Speed in pixels per frame
        """
        return self.speed