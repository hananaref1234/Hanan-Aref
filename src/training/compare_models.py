"""Compare combined vs pretrained generic detection, and custom Exp2b regression."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.dataset.taxonomy import load_dataset_yaml_names
from src.dataset.validator import IMAGE_SUFFIXES
from src.dataset.yolo_labels import parse_yolo_label_file
from src.training.matching import PredictedBox, match_predictions

GENERIC_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
)
CUSTOM_NAMES = (
    "pole",
    "bollard",
    "stairs",
    "curb",
    "barrier",
    "tree_trunk",
)
COCO_NAME_TO_COMBINED = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "chair": "chair",
}


def per_class_prf(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for name, stats in counts.items():
        tp = stats.get("true_positives", 0)
        fp = stats.get("false_positives", 0)
        fn = stats.get("false_negatives", 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        report[name] = {
            "test_instances": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        }
    return report


def evaluate_split_subset(
    model,
    dataset_root: Path,
    *,
    split: str,
    keep_names: tuple[str, ...],
    names: dict[int, str] | None = None,
    image_size: int = 640,
    confidence: float = 0.25,
    iou_threshold: float = 0.5,
    coco_name_map: bool = False,
) -> dict[str, dict[str, int]]:
    """Greedy IoU match restricted to ``keep_names``."""
    names = names or load_dataset_yaml_names(dataset_root / "dataset.yaml")
    keep = set(keep_names)
    name_to_id = {name: class_id for class_id, name in names.items()}
    keep_ids = {name_to_id[name] for name in keep if name in name_to_id}
    counts = {
        name: {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        for name in keep_names
    }
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    coco_names = None
    if coco_name_map:
        coco_names = getattr(model, "names", None) or {}
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        truth = [
            box
            for box in parse_yolo_label_file(label_path, max_class_id=max(names))
            if box.class_id in keep_ids
        ]
        if coco_name_map and not truth:
            continue
        results = model.predict(frame, imgsz=image_size, conf=confidence, verbose=False)
        predictions = _predictions_from_result(
            results[0],
            width,
            height,
            keep_ids=keep_ids if not coco_name_map else None,
            coco_names=coco_names,
            combined_names=names,
        )
        matched = match_predictions(predictions, truth, iou_threshold=iou_threshold)
        for pred, _gt in matched.true_positives:
            name = names.get(pred.class_id, str(pred.class_id))
            if name in counts:
                counts[name]["true_positives"] += 1
        for pred in matched.false_positives:
            name = names.get(pred.class_id, str(pred.class_id))
            if name in counts:
                counts[name]["false_positives"] += 1
        for gt in matched.false_negatives:
            name = names.get(gt.class_id, str(gt.class_id))
            if name in counts:
                counts[name]["false_negatives"] += 1
    return counts


def compare_generic_vs_pretrained(
    combined_model,
    pretrained_model,
    dataset_root: Path,
    output_path: Path,
    *,
    split: str = "test",
) -> dict[str, object]:
    names = load_dataset_yaml_names(dataset_root / "dataset.yaml")
    combined_counts = evaluate_split_subset(
        combined_model,
        dataset_root,
        split=split,
        keep_names=GENERIC_NAMES,
        names=names,
    )
    pretrained_counts = evaluate_split_subset(
        pretrained_model,
        dataset_root,
        split=split,
        keep_names=GENERIC_NAMES,
        names=names,
        coco_name_map=True,
    )
    combined = per_class_prf(combined_counts)
    pretrained = per_class_prf(pretrained_counts)
    delta: dict[str, dict[str, float]] = {}
    for name in GENERIC_NAMES:
        delta[name] = {
            "precision": round(
                combined[name]["precision"] - pretrained[name]["precision"], 3
            ),
            "recall": round(combined[name]["recall"] - pretrained[name]["recall"], 3),
        }
    payload = {
        "split": split,
        "note": (
            "Greedy IoU=0.5 match on combined-dataset labels. Not Ultralytics mAP. "
            "Pretrained yolo26n predictions are remapped from COCO names. "
            "Only images/boxes of the seven generic classes are scored."
        ),
        "combined": combined,
        "pretrained_yolo26n": pretrained,
        "delta_combined_minus_pretrained": delta,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compare_custom_vs_exp2b(
    combined_model,
    exp2b_root: Path,
    output_path: Path,
    *,
    split: str = "test",
) -> dict[str, object]:
    names = load_dataset_yaml_names(exp2b_root / "dataset.yaml")
    counts = evaluate_split_subset(
        combined_model,
        exp2b_root,
        split=split,
        keep_names=CUSTOM_NAMES,
        names=names,
    )
    payload = {
        "split": split,
        "dataset": str(exp2b_root),
        "note": (
            "Combined model scored on the exact Exp2b held-out custom split. "
            "Only custom-class predictions are matched; generic predictions are ignored "
            "so unlabeled people on custom images do not count as false positives."
        ),
        "combined_on_exp2b_test": per_class_prf(counts),
        "exp2b_reference": {
            "pole": {"precision": 1.000, "recall": 0.047, "mAP50": 0.045, "mAP50-95": 0.027},
            "bollard": {"precision": 0.589, "recall": 0.314, "mAP50": 0.238, "mAP50-95": 0.106},
            "stairs": {"precision": 0.902, "recall": 0.696, "mAP50": 0.677, "mAP50-95": 0.455},
            "curb": {"precision": 0.500, "recall": 0.296, "mAP50": 0.214, "mAP50-95": 0.127},
            "barrier": {"precision": 0.829, "recall": 0.716, "mAP50": 0.692, "mAP50-95": 0.507},
            "tree_trunk": {"precision": 0.640, "recall": 0.296, "mAP50": 0.267, "mAP50-95": 0.110},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _predictions_from_result(
    result: object,
    width: int,
    height: int,
    *,
    keep_ids: set[int] | None,
    coco_names: dict | None,
    combined_names: dict[int, str],
) -> list[PredictedBox]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    combined_name_to_id = {name: class_id for class_id, name in combined_names.items()}
    predictions: list[PredictedBox] = []
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        raw_id = int(box.cls[0])
        if coco_names is not None:
            coco_name = str(coco_names.get(raw_id, ""))
            mapped = COCO_NAME_TO_COMBINED.get(coco_name)
            if mapped is None:
                continue
            class_id = combined_name_to_id[mapped]
        else:
            class_id = raw_id
            if keep_ids is not None and class_id not in keep_ids:
                continue
        predictions.append(
            PredictedBox(
                class_id=class_id,
                x_center=((x1 + x2) / 2.0) / width,
                y_center=((y1 + y2) / 2.0) / height,
                width=bw / width,
                height=bh / height,
                confidence=float(box.conf[0]),
            )
        )
    return predictions
