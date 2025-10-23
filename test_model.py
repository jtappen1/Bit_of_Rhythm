import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import utils
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict

class ObjectDetectorWithVelocity:
    def __init__(self, model_path, video_path, output_dir="detection_results"):
        """
        Initialize the detector
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            video_path: Path to input video
            output_dir: Directory to save results
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tracking data
        self.track_history = defaultdict(list)  # {track_id: [(x, y, frame), ...]}
        self.velocity_data = []  # List of velocity records
        
        print(f"Video: {video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Total frames: {self.total_frames}")
        
    def calculate_velocity(self, track_id, current_pos, current_frame):
        """
        Calculate velocity based on position history
        
        Args:
            track_id: Object tracking ID
            current_pos: Current (x, y) position
            current_frame: Current frame number
            
        Returns:
            velocity_x, velocity_y, speed (in pixels/second)
        """
        history = self.track_history[track_id]
        
        if len(history) < 2:
            return 0, 0, 0
        
        # Get previous position
        prev_x, prev_y, prev_frame = history[-1]
        curr_x, curr_y = current_pos
        
        # Calculate time difference
        frame_diff = current_frame - prev_frame
        if frame_diff == 0:
            return 0, 0, 0
        
        time_diff = frame_diff / self.fps  # Time in seconds
        
        # Calculate displacement
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        
        # Calculate velocity (pixels per second)
        vx = dx / time_diff
        vy = dy / time_diff
        speed = np.sqrt(vx**2 + vy**2)
        
        return vx, vy, speed
    
    def process_video(self, conf_threshold=0.25, iou_threshold=0.45, 
                     show_video=True, save_video=True, pixels_per_meter=None):
        """
        Process video with object detection and velocity tracking
        
        Args:
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            show_video: Whether to display video during processing
            save_video: Whether to save annotated video
            pixels_per_meter: Conversion factor from pixels to meters (optional)
        """
        output_video_path = os.path.join(self.output_dir, "output_video.mp4")
        
        # Video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.fps, 
                                 (self.width, self.height))
        
        frame_num = 0
        
        print("\nProcessing video...")
        print("Press 'q' to quit early\n")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Run YOLOv8 detection first (tracking might not work with few detections)
            results = self.model(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Try to add tracking if available
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                try:
                    results = self.model.track(
                        frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        persist=True,
                        verbose=False
                    )
                except:
                    pass  # Fall back to detection without tracking
            
            # Debug: Print detection info
            if frame_num % 30 == 0:  # Print every 30 frames
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    print(f"Frame {frame_num}: {len(results[0].boxes)} detections")
                    print(f"  Confidences: {results[0].boxes.conf.cpu().numpy()}")
                    print(f"  Classes: {results[0].boxes.cls.cpu().numpy()}")
                    if results[0].boxes.id is not None:
                        print(f"  Tracked IDs: {results[0].boxes.id.cpu().numpy()}")
                    else:
                        print(f"  No tracking IDs (using detection only)")
                else:
                    print(f"Frame {frame_num}: No detections (try lowering confidence threshold)")
            
            # Process detections (with or without tracking)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Get track IDs if available, otherwise use frame-based pseudo IDs
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    # Create pseudo IDs based on position (for velocity tracking without proper tracker)
                    track_ids = list(range(len(boxes)))
                
                for box, track_id, conf, class_id in zip(boxes, track_ids, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    
                    # Calculate center point
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Calculate velocity
                    vx, vy, speed = self.calculate_velocity(track_id, (cx, cy), frame_num)
                    
                    # Update tracking history
                    self.track_history[track_id].append((cx, cy, frame_num))
                    
                    # Keep only recent history (last 30 frames)
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)
                    
                    # Convert to real-world units if calibration provided
                    if pixels_per_meter:
                        speed_real = speed / pixels_per_meter  # m/s
                        vx_real = vx / pixels_per_meter
                        vy_real = vy / pixels_per_meter
                    else:
                        speed_real = speed
                        vx_real = vx
                        vy_real = vy
                    
                    # Store velocity data
                    class_name = self.model.names[class_id]
                    self.velocity_data.append({
                        'frame': frame_num,
                        'time_sec': frame_num / self.fps,
                        'track_id': track_id,
                        'class': class_name,
                        'x': cx,
                        'y': cy,
                        'vx': vx_real,
                        'vy': vy_real,
                        'speed': speed_real,
                        'confidence': conf
                    })
                    
                    # Draw bounding box
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label with velocity
                    if pixels_per_meter:
                        label = f"ID:{track_id} {class_name} {speed_real:.2f} m/s"
                    else:
                        label = f"ID:{track_id} {class_name} {speed:.1f} px/s"
                    
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw velocity vector
                    if speed > 0:
                        end_x = int(cx + vx * 0.1)  # Scale for visualization
                        end_y = int(cy + vy * 0.1)
                        cv2.arrowedLine(frame, (int(cx), int(cy)), (end_x, end_y),
                                       (255, 0, 0), 2, tipLength=0.3)
                    
                    # Draw tracking trail
                    points = [(int(x), int(y)) for x, y, _ in self.track_history[track_id]]
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)
            
            # Add frame info
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            info_text = f"Frame: {frame_num}/{self.total_frames} | Objects: {num_detections} | Conf: {conf_threshold}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame
            if save_video:
                out.write(frame)
            
            # Display frame
            if show_video:
                cv2.imshow('Object Detection with Velocity', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Interrupted by user")
                    break
            
            frame_num += 1
            
            # Progress update
            if frame_num % 30 == 0:
                progress = (frame_num / self.total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Cleanup
        self.cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        if save_video:
            print(f"Video saved to: {output_video_path}")
        
        # Save velocity data to CSV
        self.save_velocity_data()
        
        # Generate statistics
        self.generate_statistics()
    
    def save_velocity_data(self):
        """Save velocity data to CSV"""
        csv_path = os.path.join(self.output_dir, "velocity_data.csv")
        df = pd.DataFrame(self.velocity_data)
        df.to_csv(csv_path, index=False)
        print(f"Velocity data saved to: {csv_path}")
        
    def generate_statistics(self):
        """Generate and save velocity statistics"""
        if not self.velocity_data:
            print("No velocity data to analyze")
            return
        
        df = pd.DataFrame(self.velocity_data)
        
        stats_path = os.path.join(self.output_dir, "statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("VELOCITY STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"  Total detections: {len(df)}\n")
            f.write(f"  Unique objects tracked: {df['track_id'].nunique()}\n")
            f.write(f"  Average speed: {df['speed'].mean():.2f}\n")
            f.write(f"  Max speed: {df['speed'].max():.2f}\n")
            f.write(f"  Min speed: {df['speed'].min():.2f}\n\n")
            
            # Per-class statistics
            f.write("Per-Class Statistics:\n")
            for class_name in df['class'].unique():
                class_df = df[df['class'] == class_name]
                f.write(f"\n  {class_name}:\n")
                f.write(f"    Count: {len(class_df)}\n")
                f.write(f"    Avg speed: {class_df['speed'].mean():.2f}\n")
                f.write(f"    Max speed: {class_df['speed'].max():.2f}\n")
            
            # Per-object statistics
            f.write("\n\nPer-Object Statistics:\n")
            for track_id in sorted(df['track_id'].unique()):
                obj_df = df[df['track_id'] == track_id]
                class_name = obj_df['class'].iloc[0]
                f.write(f"\n  ID {track_id} ({class_name}):\n")
                f.write(f"    Frames tracked: {len(obj_df)}\n")
                f.write(f"    Duration: {obj_df['time_sec'].max() - obj_df['time_sec'].min():.2f}s\n")
                f.write(f"    Avg speed: {obj_df['speed'].mean():.2f}\n")
                f.write(f"    Max speed: {obj_df['speed'].max():.2f}\n")
        
        print(f"Statistics saved to: {stats_path}")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total detections: {len(df)}")
        print(f"Unique objects: {df['track_id'].nunique()}")
        print(f"Average speed: {df['speed'].mean():.2f}")
        print(f"Max speed: {df['speed'].max():.2f}")
        print("=" * 60)

# Usage
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "runs/detect/custom_model/weights/best.pt"  # Your trained model
    VIDEO_PATH = utils.select_video_file()  # Video to process
    OUTPUT_DIR = "detection_results"
    
    # Optional: Calibration for real-world measurements
    # If you know the real-world scale, set pixels_per_meter
    # Example: If 100 pixels = 1 meter, then pixels_per_meter = 100
    PIXELS_PER_METER = None  # Set to None for pixel measurements
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.05  # Lowered significantly to catch more detections
    IOU_THRESHOLD = 0.45
    
    # Create detector
    detector = ObjectDetectorWithVelocity(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Process video
    detector.process_video(
        conf_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        show_video=True,  # Display video while processing
        save_video=True,  # Save annotated video
        pixels_per_meter=PIXELS_PER_METER
    )
    
    print("\n✅ Complete! Check the output directory for results.")