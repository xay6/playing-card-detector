from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # smallest model for fastest training

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,     # GPU
        workers=0,    # REQUIRED on Windows
        project="runs_cards",
        name="cards_detector"
    )

if __name__ == "__main__":
    main()
