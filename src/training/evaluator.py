"""Evaluation helpers for a custom Li-Stick detector."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.dataset.taxonomy import SAFETY_CRITICAL_CLASSES, load_dataset_yaml_names
from src.dataset.split_report import count_split_classes
from src.dataset.validator import IMAGE_SUFFIXES
from src.dataset.yolo_labels import YoloBox, parse_yolo_label_file
from src.training.matching import PredictedBox, match_predictions
from src.training.trainer import TrainingNotReadyError, ensure_dataset_ready

logger = logging.getLogger(__name__)

_MAX_EXAMPLES_PER_BUCKET = 20


@dataclass
class ClassMetrics:
    name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0


@dataclass
class ErrorIndex:
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)


def evaluate_custom_detector(
    model_path: Path,
    dataset_yaml: Path,
    output_dir: Path,
    image_size: int = 640,
    confidence: float = 0.25,
    iou_threshold: float = 0.5,
    split: str = "test",
) -> Path:
    """Run Ultralytics val() plus a per-class report and visual examples."""
    ensure_dataset_ready(dataset_yaml)
    if not model_path.exists():
        raise TrainingNotReadyError(
            f"Custom model not found: {model_path}. "
            "TRAINING: NOT STARTED. Train after a labeled dataset exists."
        )
    if split not in {"train", "val", "test"}:
        raise TrainingNotReadyError(f"Unsupported eval split {split!r}. Use train, val, or test.")

    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    names = load_dataset_yaml_names(dataset_yaml)
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(dataset_yaml),
        imgsz=image_size,
        conf=confidence,
        iou=iou_threshold,
        split=split,
        project=str(output_dir),
        name="ultralytics_val",
        exist_ok=True,
        plots=True,
    )
    summary = _ultralytics_summary(metrics, names)
    summary["split"] = split
    summary["model_path"] = str(model_path)
    summary["dataset_yaml"] = str(dataset_yaml)
    summary["confidence"] = confidence
    summary["iou_threshold"] = iou_threshold
    summary["test_instances"] = _label_instance_counts(
        dataset_yaml.parent, split=split, names=names
    )
    error_index = _export_visual_examples(
        model=model,
        dataset_root=dataset_yaml.parent,
        output_dir=output_dir / "evaluation",
        names=names,
        image_size=image_size,
        confidence=confidence,
        iou_threshold=iou_threshold,
        split=split,
    )
    summary["match_counts"] = error_index.counts
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "error_index.json").write_text(
        json.dumps({"counts": error_index.counts, "examples": error_index.examples}, indent=2),
        encoding="utf-8",
    )
    _write_eval_report(output_dir / "evaluation_report.md", summary, names)
    _write_safety_report(output_dir / "safety_critical_report.md", summary, names)
    _write_error_analysis_stub(output_dir / "ERROR_ANALYSIS.md", summary, names)
    return output_dir


def _label_instance_counts(
    dataset_root: Path,
    *,
    split: str,
    names: dict[int, str],
) -> dict[str, int]:
    counts = count_split_classes(dataset_root)
    per_split = counts.get("per_class_by_split", {})
    if not isinstance(per_split, dict):
        return {}
    split_counts = per_split.get(split, {})
    if not isinstance(split_counts, dict):
        return {}
    return {name: int(split_counts.get(name, 0)) for name in names.values()}


def _ultralytics_summary(metrics: object, names: dict[int, str]) -> dict[str, object]:
    box = getattr(metrics, "box", None)
    display_names = getattr(metrics, "names", None) or names
    if not isinstance(display_names, dict):
        display_names = names
    summary: dict[str, object] = {
        "precision": _metric_attr(box, "mp"),
        "recall": _metric_attr(box, "mr"),
        "mAP50": _metric_attr(box, "map50"),
        "mAP50-95": _metric_attr(box, "map"),
        "per_class": {},
        "confusion_matrix": _confusion_matrix_payload(metrics, display_names),
    }
    per_class: dict[str, dict[str, float | None]] = {}
    precision = _as_float_list(getattr(box, "p", None) if box is not None else None)
    recall = _as_float_list(getattr(box, "r", None) if box is not None else None)
    maps = _as_float_list(getattr(box, "maps", None) if box is not None else None)
    ap50 = _per_class_ap50(box)
    class_ids = sorted(display_names)
    for class_id in class_ids:
        name = str(display_names.get(class_id, names.get(class_id, class_id)))
        stats: dict[str, float | None] = {}
        if precision is not None and class_id < len(precision):
            stats["precision"] = precision[class_id]
        if recall is not None and class_id < len(recall):
            stats["recall"] = recall[class_id]
        if ap50 is not None and class_id < len(ap50):
            stats["mAP50"] = ap50[class_id]
        if maps is not None and class_id < len(maps):
            stats["mAP50-95"] = maps[class_id]
        per_class[name] = stats
    summary["per_class"] = per_class
    return summary


def _per_class_ap50(box: object) -> list[float] | None:
    if box is None:
        return None
    ap50 = getattr(box, "ap50", None)
    values = _as_float_list(ap50)
    if values is not None:
        return values
    all_ap = getattr(box, "all_ap", None)
    if all_ap is None:
        return None
    array = np.asarray(all_ap)
    if array.ndim == 2 and array.shape[1] >= 1:
        return [float(v) for v in array[:, 0].tolist()]
    return _as_float_list(all_ap)


def _confusion_matrix_payload(metrics: object, names: dict[int, str]) -> dict[str, object] | None:
    cm = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(cm, "matrix", None) if cm is not None else None
    if matrix is None:
        return None
    array = np.asarray(matrix)
    class_names = [str(names[i]) for i in sorted(names)]
    class_names.append("background")
    return {
        "note": (
            "Ultralytics ConfusionMatrix including a background row/column. "
            "Inspect axes before interpreting; values are dumped raw."
        ),
        "labels": class_names,
        "matrix": array.tolist(),
    }


def _as_float_list(value: object) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[assignment]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                out.append(float(item))
            else:
                return None
        return out
    return None


def _metric_attr(box: object, name: str) -> float | None:
    if box is None:
        return None
    value = getattr(box, name, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _write_eval_report(
    path: Path,
    summary: dict[str, object],
    names: dict[int, str],
) -> None:
    per_class = summary.get("per_class") or {}
    lines = [
        "# Evaluation report",
        "",
        "Held-out split metrics from Ultralytics `val()`.",
        "This is not a safety certification. Direction (up/down) is not evaluated.",
        "",
        f"Split: {summary.get('split')}",
        f"Overall precision: {summary.get('precision')}",
        f"Overall recall: {summary.get('recall')}",
        f"mAP50: {summary.get('mAP50')}",
        f"mAP50-95: {summary.get('mAP50-95')}",
        "",
    ]
    for class_id in sorted(names):
        name = names[class_id]
        stats = per_class.get(name, {}) if isinstance(per_class, dict) else {}
        instances = (summary.get("test_instances") or {}).get(name, "not available")
        lines.extend(
            [
                f"## {name}",
                f"- test instances: {instances}",
                f"- precision: {stats.get('precision', 'not available')}",
                f"- recall: {stats.get('recall', 'not available')}",
                f"- mAP50: {stats.get('mAP50', 'not available')}",
                f"- mAP50-95: {stats.get('mAP50-95', 'not available')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_safety_report(
    path: Path,
    summary: dict[str, object],
    names: dict[int, str],
) -> None:
    trained = set(names.values())
    critical = [name for name in SAFETY_CRITICAL_CLASSES if name in trained]
    lines = [
        "# Safety-critical class report",
        "",
        "False negatives on drop-off and overhead obstacles are especially costly.",
        "Do not treat a high overall mAP as navigation readiness.",
        "",
    ]
    if not critical:
        lines.extend(
            [
                "This training run did **not** include the safety-critical classes "
                f"{', '.join(SAFETY_CRITICAL_CLASSES)}.",
                "No drop_off or low_overhead_obstacle metrics are available.",
                "",
            ]
        )
    else:
        per_class = summary.get("per_class") or {}
        for name in critical:
            stats = per_class.get(name, {}) if isinstance(per_class, dict) else {}
            lines.extend(
                [
                    f"## {name}",
                    f"- mAP50-95: {stats.get('mAP50-95', 'not available')}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_error_analysis_stub(
    path: Path,
    summary: dict[str, object],
    names: dict[int, str],
) -> None:
    counts = summary.get("match_counts") or {}
    lines = [
        "# Error analysis (quantitative)",
        "",
        "Greedy IoU matching on the held-out split (conf/IoU as in metrics.json).",
        "Visual examples are under `evaluation/false_positive/` and `evaluation/false_negative/`.",
        "Do not invent qualitative failure modes from this table alone.",
        "",
        f"Split: {summary.get('split')}",
        "",
    ]
    if isinstance(counts, dict):
        for name in [names[i] for i in sorted(names)]:
            stats = counts.get(name, {})
            lines.append(
                f"- **{name}**: TP={stats.get('true_positives', 0)} "
                f"FP={stats.get('false_positives', 0)} "
                f"FN={stats.get('false_negatives', 0)}"
            )
        lines.append("")
    cm = summary.get("confusion_matrix")
    if isinstance(cm, dict) and cm.get("matrix") is not None:
        lines.extend(
            [
                "## Confusion matrix (raw Ultralytics dump)",
                "",
                f"Labels: {cm.get('labels')}",
                "",
                "```",
                json.dumps(cm.get("matrix"), indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_visual_examples(
    model: object,
    dataset_root: Path,
    output_dir: Path,
    names: dict[int, str],
    image_size: int,
    confidence: float,
    iou_threshold: float,
    split: str,
) -> ErrorIndex:
    for folder in ("true_positive", "false_positive", "false_negative"):
        for name in names.values():
            (output_dir / folder / name).mkdir(parents=True, exist_ok=True)

    max_class_id = max(names) if names else -1
    saved = defaultdict(int)
    counts: dict[str, dict[str, int]] = {
        name: {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        for name in names.values()
    }
    examples: dict[str, list[str]] = defaultdict(list)
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.exists():
        return ErrorIndex(counts=counts, examples=dict(examples))

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
        truth = parse_yolo_label_file(label_path, max_class_id=max_class_id)
        results = model.predict(frame, imgsz=image_size, conf=confidence, verbose=False)
        predictions = _predictions_from_result(results[0], width, height)
        matched = match_predictions(predictions, truth, iou_threshold=iou_threshold)

        for pred, _gt in matched.true_positives:
            name = names.get(pred.class_id, str(pred.class_id))
            counts[name]["true_positives"] += 1
        for pred in matched.false_positives:
            name = names.get(pred.class_id, str(pred.class_id))
            counts.setdefault(name, {"true_positives": 0, "false_positives": 0, "false_negatives": 0})
            counts[name]["false_positives"] += 1
            key = f"false_positive/{name}"
            if saved[key] < _MAX_EXAMPLES_PER_BUCKET:
                annotated = _annotate(frame, fp=[pred], names=names)
                dest = output_dir / "false_positive" / name / f"{split}_{image_path.stem}_{saved[key]}.jpg"
                cv2.imwrite(str(dest), annotated)
                examples[key].append(str(dest))
                saved[key] += 1
        for gt in matched.false_negatives:
            name = names.get(gt.class_id, str(gt.class_id))
            counts.setdefault(name, {"true_positives": 0, "false_positives": 0, "false_negatives": 0})
            counts[name]["false_negatives"] += 1
            key = f"false_negative/{name}"
            if saved[key] < _MAX_EXAMPLES_PER_BUCKET:
                annotated = _annotate(frame, fn=[gt], names=names)
                dest = output_dir / "false_negative" / name / f"{split}_{image_path.stem}_{saved[key]}.jpg"
                cv2.imwrite(str(dest), annotated)
                examples[key].append(str(dest))
                saved[key] += 1
        if matched.true_positives:
            name = names.get(matched.true_positives[0][0].class_id, "unknown")
            key = f"true_positive/{name}"
            if saved[key] < _MAX_EXAMPLES_PER_BUCKET:
                annotated = _annotate(
                    frame,
                    tp=[item[0] for item in matched.true_positives],
                    names=names,
                )
                dest = output_dir / "true_positive" / name / f"{split}_{image_path.stem}.jpg"
                cv2.imwrite(str(dest), annotated)
                examples[key].append(str(dest))
                saved[key] += 1

    return ErrorIndex(counts=counts, examples=dict(examples))


def _annotate(
    frame: np.ndarray,
    names: dict[int, str],
    tp: list[PredictedBox] | None = None,
    fp: list[PredictedBox] | None = None,
    fn: list[YoloBox] | None = None,
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    for pred in tp or []:
        _draw_yolo_box(image, pred, width, height, (40, 180, 40), f"TP {names.get(pred.class_id, pred.class_id)}")
    for pred in fp or []:
        _draw_yolo_box(image, pred, width, height, (40, 40, 220), f"FP {names.get(pred.class_id, pred.class_id)}")
    for gt in fn or []:
        _draw_yolo_box(image, gt, width, height, (220, 180, 40), f"FN {names.get(gt.class_id, gt.class_id)}")
    return image


def _draw_yolo_box(
    image: np.ndarray,
    box: PredictedBox | YoloBox,
    width: int,
    height: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1 = int(round((box.x_center - box.width / 2.0) * width))
    y1 = int(round((box.y_center - box.height / 2.0) * height))
    x2 = int(round((box.x_center + box.width / 2.0) * width))
    y2 = int(round((box.y_center + box.height / 2.0) * height))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _predictions_from_result(result: object, width: int, height: int) -> list[PredictedBox]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    predictions: list[PredictedBox] = []
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        predictions.append(
            PredictedBox(
                class_id=int(box.cls[0]),
                x_center=((x1 + x2) / 2.0) / width,
                y_center=((y1 + y2) / 2.0) / height,
                width=bw / width,
                height=bh / height,
                confidence=float(box.conf[0]),
            )
        )
    return predictions
