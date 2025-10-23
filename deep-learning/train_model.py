from ultralytics import YOLO
import cv2
import numpy as np

def train_model():
    model = YOLO("yolov8n.pt")
    print("Loaded model")
    model.train(data="data/BoR.v2-batch_2.yolov8/data.yaml", epochs=30, imgsz=640, batch=16)
    metrics = model.val()

    print(f"mAP@0.5: {metrics.box.map50}")
    print(f"mAP@0.5:0.95: {metrics.box.map}")

def inference():
    pose_estimator = YOLO('yolov8n-pose.pt')
    model = YOLO("/Users/jtappen/Projects/Bit_of_Rhythm/deep-learning/shared_weights/best.pt")  

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True) 

        poses = pose_estimator(frame)

        left_wrist_coords: tuple | None  = None
        right_wrist_coords: tuple | None = None

        for pose in poses:
            if pose.keypoints:
                keypoints = pose.keypoints.xy.cpu().numpy()
                confs = pose.keypoints.conf.cpu().numpy()
            
                lx, ly = map(int, keypoints[0][9])
                rx, ry = map(int, keypoints[0][10])
                l_conf = confs[0][9]
                r_conf = confs[0][10]
                if l_conf > 0.5:
                    cv2.circle(frame, center=(lx, ly), radius=20, color= (0, 0 , 255), thickness=-1)
                    left_wrist_coords = (lx, ly)
                    
                if r_conf > 0.5:
                    cv2.circle(frame, center=(rx, ry), radius=20, color= (0, 255 , 0), thickness=-1)
                    right_wrist_coords = (rx, ry)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                x_center = (x1 + x2) / 2
                y_center = (y1 - y2) / 2
                center_coords = np.array([x_center, y_center])
                left_dist = np.linalg.norm(left_wrist_coords - center_coords) if left_wrist_coords else None
                right_dist = np.linalg.norm(right_wrist_coords - center_coords) if right_wrist_coords else None
                if left_dist  and right_dist and left_dist < right_dist:
                    color = (0, 0, 255)
                elif left_dist  and right_dist and left_dist > right_dist:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("YOLOv8 Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    train_model()