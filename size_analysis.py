from ultralytics import YOLO
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\NAITIK\Desktop\clg\mo\MAR20_FULL")

MODEL_PATH = ROOT / "runs" / "yolo11s_20" / "weights" / "best.pt"
IMAGE_DIR = ROOT / "MAR20_YOLO" / "images" / "val"
LABEL_DIR = ROOT / "MAR20_YOLO" / "labels" / "val"
OUTPUT_FILE = ROOT / "object_size_analysis.csv"

CLASS_NAMES = [
    "A1", "A2", "A3", "A4", "A5",
    "A6", "A7", "A8", "A9", "A10",
    "A11", "A12", "A13", "A14", "A15",
    "A16", "A17", "A18", "A19", "A20"
]


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0.0

    return inter / union


def load_ground_truth(label_path, img_w, img_h):
    gt = []

    if not label_path.exists():
        return gt

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            box = yolo_to_xyxy(
                xc, yc, w, h,
                img_w, img_h
            )

            gt.append({
                "cls": cls_id,
                "box": box,
                "norm_area": w * h,
                "norm_width": w,
                "norm_height": h
            })

    return gt


def main():

    model = YOLO(str(MODEL_PATH))

    records = []

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    print(f"Analysing {len(image_paths)} validation images...")

    # -------------------------------------------------
    # PASS 1: collect all object sizes
    # -------------------------------------------------

    all_areas = []

    for image_path in image_paths:

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        from PIL import Image
        image = Image.open(image_path)
        img_w, img_h = image.size

        ground_truth = load_ground_truth(
            label_path,
            img_w,
            img_h
        )

        for gt in ground_truth:
            all_areas.append(gt["norm_area"])

    if not all_areas:
        print("No ground-truth objects found.")
        return

    # Dataset-relative size thresholds
    small_threshold = np.percentile(all_areas, 33.33)
    large_threshold = np.percentile(all_areas, 66.67)

    print("\nDataset-relative object size thresholds:")
    print(f"Small:   area <= {small_threshold:.6f}")
    print(f"Medium:  {small_threshold:.6f} - {large_threshold:.6f}")
    print(f"Large:   area > {large_threshold:.6f}")

    # -------------------------------------------------
    # PASS 2: evaluate every object
    # -------------------------------------------------

    for index, image_path in enumerate(image_paths, start=1):

        from PIL import Image
        image = Image.open(image_path)
        img_w, img_h = image.size

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        ground_truth = load_ground_truth(
            label_path,
            img_w,
            img_h
        )

        if not ground_truth:
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            device=0,
            verbose=False
        )[0]

        predictions = []

        if result.boxes is not None:

            for box, cls, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy()
            ):

                predictions.append({
                    "cls": int(cls),
                    "conf": float(conf),
                    "box": box.tolist()
                })

        used_predictions = set()

        for gt in ground_truth:

            best_iou = 0
            best_prediction = None
            best_index = None

            for pred_index, pred in enumerate(predictions):

                if pred_index in used_predictions:
                    continue

                current_iou = box_iou(
                    gt["box"],
                    pred["box"]
                )

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_prediction = pred
                    best_index = pred_index

            # Determine object size
            if gt["norm_area"] <= small_threshold:
                size_category = "Small"
            elif gt["norm_area"] <= large_threshold:
                size_category = "Medium"
            else:
                size_category = "Large"

            if best_prediction is None or best_iou < 0.50:

                status = "Missed"

                predicted_class = "None"
                confidence = 0

            else:

                used_predictions.add(best_index)

                confidence = best_prediction["conf"]
                predicted_class = CLASS_NAMES[
                    best_prediction["cls"]
                ]

                if best_prediction["cls"] == gt["cls"]:
                    status = "Correct"
                else:
                    status = "Wrong_Class"

            records.append({
                "image": image_path.name,
                "true_class": CLASS_NAMES[gt["cls"]],
                "size": size_category,
                "norm_area": gt["norm_area"],
                "width_fraction": gt["norm_width"],
                "height_fraction": gt["norm_height"],
                "status": status,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "best_iou": best_iou
            })

        if index % 25 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    # -------------------------------------------------
    # SAVE OBJECT-LEVEL DATA
    # -------------------------------------------------

    df = pd.DataFrame(records)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("OBJECT-SIZE FAILURE ANALYSIS")
    print("=" * 60)

    summary = (
        df.groupby("size")
        .agg(
            Objects=("status", "count"),
            Correct=("status", lambda x: (x == "Correct").sum()),
            Wrong_Class=("status", lambda x: (x == "Wrong_Class").sum()),
            Missed=("status", lambda x: (x == "Missed").sum())
        )
    )

    summary["Detection_Rate"] = (
        (summary["Correct"] + summary["Wrong_Class"])
        / summary["Objects"]
    )

    summary["Correct_Classification_Rate"] = (
        summary["Correct"] / summary["Objects"]
    )

    summary["Miss_Rate"] = (
        summary["Missed"] / summary["Objects"]
    )

    print("\n")
    print(summary.round(4))

    print("\n" + "=" * 60)
    print("Overall")
    print("=" * 60)

    total = len(df)
    correct = (df["status"] == "Correct").sum()
    wrong = (df["status"] == "Wrong_Class").sum()
    missed = (df["status"] == "Missed").sum()

    print(f"Total objects:          {total}")
    print(f"Correct:                {correct}")
    print(f"Wrong class:            {wrong}")
    print(f"Missed:                 {missed}")
    print(f"Detection rate:         {(correct + wrong) / total:.4f}")
    print(f"Classification rate:    {correct / total:.4f}")
    print(f"Miss rate:              {missed / total:.4f}")

    print("\nCSV saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()