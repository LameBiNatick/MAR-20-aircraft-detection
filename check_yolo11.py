from ultralytics import YOLO

def main():
    model = YOLO(
        r"C:\Users\NAITIK\Desktop\clg\mo\MAR20_FULL\runs\yolo11s_20\weights\best.pt"
    )

    model.val(
        data=r"C:\Users\NAITIK\Desktop\clg\mo\MAR20_FULL\MAR20_YOLO\mar20.yaml",
        split="val",
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        verbose=True
    )

if __name__ == "__main__":
    main()