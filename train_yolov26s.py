from ultralytics import YOLO


def main():
    model = YOLO("yolo26s.pt")

    model.train(
        data=r"C:\Users\NAITIK\Desktop\clg\mo\MAR20_FULL\MAR20_YOLO\mar20.yaml",
        epochs=20,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=r"C:\Users\NAITIK\Desktop\clg\mo\MAR20_FULL\runs",
        name="yolo26s_20",
        seed=42
    )


if __name__ == "__main__":
    main()