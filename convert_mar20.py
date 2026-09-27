import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

SOURCE = Path(r"C:\Users\NAITIK\Desktop\clg\mo\MAR20_FULL\MAR20")
OUTPUT = Path(r"C:\Users\NAITIK\Desktop\clg\mo\MAR20_FULL\MAR20_YOLO")

IMAGE_DIR = SOURCE / "JPEGImages"
ANNOTATION_DIR = SOURCE / "Annotations" / "Horizontal Bounding Boxes"
IMAGESETS_DIR = SOURCE / "ImageSets" / "Main"

CLASSES = [f"A{i}" for i in range(1, 21)]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

VAL_RATIO = 0.20
SEED = 42


def read_split(filename):
    path = IMAGESETS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def convert_annotation(image_id, output_label_path):
    xml_path = ANNOTATION_DIR / f"{image_id}.xml"
    image_path = IMAGE_DIR / f"{image_id}.jpg"

    if not xml_path.exists():
        raise FileNotFoundError(f"Missing annotation: {xml_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    root = ET.parse(xml_path).getroot()

    # Use the actual image dimensions.
    # Some MAR20 XML files contain incorrect 0x0 dimensions.
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    labels = []

    for obj in root.findall("object"):
        class_element = obj.find("name")

        if class_element is None or class_element.text is None:
            raise ValueError(
                f"Missing class name in annotation: {xml_path}"
            )

        class_name = class_element.text.strip()

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown class '{class_name}' in {xml_path}"
            )

        box = obj.find("bndbox")

        if box is None:
            raise ValueError(
                f"Missing bounding box in {xml_path}"
            )

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        # Basic validation
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                f"Invalid bounding box in {xml_path}: "
                f"{xmin}, {ymin}, {xmax}, {ymax}"
            )

        # Convert Pascal VOC:
        # xmin, ymin, xmax, ymax
        #
        # to YOLO:
        # x_center, y_center, width, height

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height

        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        # Make sure normalized values remain valid
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)

        class_id = CLASS_TO_ID[class_name]

        labels.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    output_label_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_label_path.write_text(
        "\n".join(labels)
    )


def copy_split(image_ids, split):
    image_output = OUTPUT / "images" / split
    label_output = OUTPUT / "labels" / split

    image_output.mkdir(
        parents=True,
        exist_ok=True
    )

    label_output.mkdir(
        parents=True,
        exist_ok=True
    )

    total = len(image_ids)

    for i, image_id in enumerate(image_ids, start=1):

        image_path = IMAGE_DIR / f"{image_id}.jpg"
        label_path = label_output / f"{image_id}.txt"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        # Copy image
        shutil.copy2(
            image_path,
            image_output / image_path.name
        )

        # Convert XML → YOLO TXT
        convert_annotation(
            image_id,
            label_path
        )

        if i % 100 == 0 or i == total:
            print(f"{split}: {i}/{total}")


def main():

    print("========================================")
    print("       MAR20 → YOLO CONVERTER")
    print("========================================")

    print("\nChecking dataset paths...")

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source directory not found:\n{SOURCE}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not ANNOTATION_DIR.exists():
        raise FileNotFoundError(
            f"Annotation directory not found:\n{ANNOTATION_DIR}"
        )

    if not IMAGESETS_DIR.exists():
        raise FileNotFoundError(
            f"ImageSets directory not found:\n{IMAGESETS_DIR}"
        )

    print("Dataset paths OK.")

    print("\nReading official MAR20 splits...")

    train_ids = read_split("train.txt")
    test_ids = read_split("test.txt")

    print(f"Official train images: {len(train_ids)}")
    print(f"Official test images:  {len(test_ids)}")

    # Check for duplicate IDs
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Duplicate image IDs found in train.txt")

    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate image IDs found in test.txt")

    # Make sure train and test don't overlap
    overlap = set(train_ids) & set(test_ids)

    if overlap:
        raise ValueError(
            f"Train/test overlap detected: {len(overlap)} images"
        )

    # Split official training images into:
    # 80% train
    # 20% validation
    random.seed(SEED)
    random.shuffle(train_ids)

    val_size = int(len(train_ids) * VAL_RATIO)

    val_ids = train_ids[:val_size]
    train_ids = train_ids[val_size:]

    print(f"Training images:       {len(train_ids)}")
    print(f"Validation images:     {len(val_ids)}")
    print(f"Test images:           {len(test_ids)}")

    # Remove old output to avoid stale/partial files
    if OUTPUT.exists():
        print("\nRemoving previous incomplete YOLO dataset...")
        shutil.rmtree(OUTPUT)

    # Convert datasets
    print("\n========================================")
    print("Converting training set...")
    print("========================================")

    copy_split(train_ids, "train")

    print("\n========================================")
    print("Converting validation set...")
    print("========================================")

    copy_split(val_ids, "val")

    print("\n========================================")
    print("Converting test set...")
    print("========================================")

    copy_split(test_ids, "test")

    # Create dataset YAML
    yaml_content = f"""path: {OUTPUT.as_posix()}

train: images/train
val: images/val
test: images/test

names:
"""

    for i, class_name in enumerate(CLASSES):
        yaml_content += f"  {i}: {class_name}\n"

    yaml_path = OUTPUT / "mar20.yaml"
    yaml_path.write_text(yaml_content)

    print("\n========================================")
    print("      DATASET CONVERSION COMPLETE")
    print("========================================")

    print(f"\nYOLO dataset:")
    print(OUTPUT)

    print(f"\nYAML file:")
    print(yaml_path)

    print("\nClass mapping:")

    for class_name, class_id in CLASS_TO_ID.items():
        print(f"{class_id}: {class_name}")


if __name__ == "__main__":
    main()