import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
from tkinter import filedialog


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def select_video_file():
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        )
    )
    
    return file_path

# The train_model function is kept, but it's not called in the main execution block
# if we switch to inference.
def train_model():
    model = YOLO("yolov8n.pt")
    print("Loaded model")
    model.train(data="deep-learning\\data\\BoR.v2-batch_2.yolov8\\data.yaml", epochs=30, imgsz=640, batch=16)
    metrics = model.val()

    print(f"mAP@0.5: {metrics.box.map50}")
    print(f"mAP@0.5:0.95: {metrics.box.map}")

def inference():
    pose_estimator = YOLO('yolov8n-pose.pt')
    model = YOLO("runs\\detect\\train4\weights\\best.pt")

    '''cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return'''
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
        poses = pose_estimator(frame, verbose=False) # verbose=False to clean up console

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

        results = model(frame, stream=True, verbose=False) 
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls)
                label = model.names[cls_id]

                # Calculate current center coordinates
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2 
                center_coords = np.array([x_center, y_center])
                
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
                
                color = (255, 255, 255) 
                left_dist = np.linalg.norm(left_wrist_coords - center_coords) if left_wrist_coords is not None else float('inf')
                right_dist = np.linalg.norm(right_wrist_coords - center_coords) if right_wrist_coords is not None else float('inf')
                
                # Only apply color if at least one wrist is detected
                if left_wrist_coords is not None or right_wrist_coords is not None:
                    if left_dist < right_dist:
                        color = (0, 0, 255) 
                    elif right_dist < left_dist:
                        color = (0, 255, 0)

                label_text = f"{label} Conf:{conf:.2f} Vel:{velocity:.2f} p/s"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # --- 3. Display and Exit (Unchanged) ---
        cv2.imshow("YOLOv8 Live", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # If the model is already trained, you should comment out train_model()
    # and call inference() directly.
    # train_model() 
    inference()