from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("runs_cards/cards_detector2/weights/best.pt")

    source = "test/images"     # folder or image

    model.predict(
        source=source,
        conf=0.25,
        save=True,
        device=0,
        project="runs_cards",
        name="predictions",
        workers=0
    )

    print("Done! Check:", Path("runs_cards/predictions").resolve())

if __name__ == "__main__":
    main()
