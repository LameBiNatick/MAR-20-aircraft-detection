from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


ROOT = Path(r"C:\Users\NAITIK\Desktop\clg\mo\MAR20_FULL")

MODEL_PATH = ROOT / "runs" / "yolo11s_20" / "weights" / "best.pt"
IMAGE_DIR = ROOT / "MAR20_YOLO" / "images" / "val"
LABEL_DIR = ROOT / "MAR20_YOLO" / "labels" / "val"
OUTPUT_DIR = ROOT / "failure_analysis"


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


def iou(box1, box2):
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
        return 0

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
                "box": box
            })

    return gt


def draw_box(draw, box, text, color, width=3):
    draw.rectangle(box, outline=color, width=width)

    x1, y1, _, _ = box

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    draw.text(
        (x1, max(0, y1 - 20)),
        text,
        fill=color,
        font=font
    )


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    for folder in [
        "A1_to_A19",
        "A19_to_A1",
        "A13_to_A15",
        "A15_to_A13",
        "missed"
    ]:
        (OUTPUT_DIR / folder).mkdir(exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    saved = {
        "A1_to_A19": 0,
        "A19_to_A1": 0,
        "A13_to_A15": 0,
        "A15_to_A13": 0,
        "missed": 0
    }

    max_per_category = 5

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    print(f"Analysing {len(image_paths)} validation images...")

    for index, image_path in enumerate(image_paths, start=1):

        image = Image.open(image_path).convert("RGB")
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

        matched_predictions = set()

        found_categories = set()

        # Match every ground-truth object to its best prediction
        for gt_index, gt in enumerate(ground_truth):

            best_iou = 0
            best_pred_index = None

            for pred_index, pred in enumerate(predictions):

                if pred_index in matched_predictions:
                    continue

                current_iou = iou(gt["box"], pred["box"])

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred_index = pred_index

            # Missed detection
            if best_pred_index is None or best_iou < 0.50:

                if saved["missed"] >= max_per_category:
                    continue

                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)

                draw_box(
                    draw,
                    gt["box"],
                    f"GT: {CLASS_NAMES[gt['cls']]}",
                    "red"
                )

                output_path = (
                    OUTPUT_DIR /
                    "missed" /
                    f"{image_path.stem}_missed.jpg"
                )

                annotated.save(output_path)

                saved["missed"] += 1
                found_categories.add("missed")

                continue

            pred = predictions[best_pred_index]
            matched_predictions.add(best_pred_index)

            true_class = CLASS_NAMES[gt["cls"]]
            predicted_class = CLASS_NAMES[pred["cls"]]

            # Ignore correct predictions
            if true_class == predicted_class:
                continue

            category = f"{true_class}_to_{predicted_class}"

            allowed_categories = {
                "A1_to_A19",
                "A19_to_A1",
                "A13_to_A15",
                "A15_to_A13"
            }

            if category not in allowed_categories:
                continue

            if saved[category] >= max_per_category:
                continue

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            # Ground truth
            draw_box(
                draw,
                gt["box"],
                f"GT: {true_class}",
                "red"
            )

            # Prediction
            draw_box(
                draw,
                pred["box"],
                f"Pred: {predicted_class} ({pred['conf']:.2f})",
                "blue"
            )

            output_path = (
                OUTPUT_DIR /
                category /
                f"{image_path.stem}_{category}.jpg"
            )

            annotated.save(output_path)

            saved[category] += 1
            found_categories.add(category)

        if index % 25 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    print("\nDONE")
    print("\nSaved examples:")

    for category, count in saved.items():
        print(f"{category}: {count}")

    print(f"\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()