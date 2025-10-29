from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")
    print("Loaded model")
    model.train(data="deep-learning\\data\\BoR.v2-batch_2.yolov8\\data.yaml", epochs=30, imgsz=640, batch=16)
    metrics = model.val()

    print(f"mAP@0.5: {metrics.box.map50}")
    print(f"mAP@0.5:0.95: {metrics.box.map}")

if __name__ == "__main__":
    train_model()