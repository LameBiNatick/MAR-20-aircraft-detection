# MAR-20-aircraft-detection

# Fine-Grained Military Aircraft Detection on MAR20

YOLO-based detection and fine-grained recognition of military aircraft
in remote sensing imagery using the MAR20 benchmark.

## Dataset

MAR20 contains 20 military aircraft classes.

The dataset was converted from the original annotations into YOLO format.

## Models Evaluated

- YOLOv5s
- YOLOv8s
- YOLO11s
- YOLO26s

## 20-Epoch Screening Results

| Model | Precision | Recall | mAP50 | mAP50-95 |
| YOLOv5s | 0.751 | 0.755 | 0.801 | 0.599 |
| YOLOv8s | 0.790 | 0.750 | 0.820 | 0.619 |
| YOLO11s | 0.801 | 0.788 | 0.833 | 0.626 |
| YOLO26s | 0.795 | 0.786 | 0.836 | 0.628 |

## Failure Analysis

Analysis includes:

- normalized confusion matrices
- per-class metrics
- fine-grained class confusion
- missed detections
- visual failure examples

## Current Research Direction

Investigating the causes of fine-grained aircraft confusion and
missed detections in remote sensing imagery.
