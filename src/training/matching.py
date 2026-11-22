"""Prediction-to-ground-truth matching for visual evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from src.dataset.yolo_labels import YoloBox


@dataclass(frozen=True)
class PredictedBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float


@dataclass(frozen=True)
class MatchResult:
    true_positives: list[tuple[PredictedBox, YoloBox]]
    false_positives: list[PredictedBox]
    false_negatives: list[YoloBox]


def yolo_iou(a: PredictedBox | YoloBox, b: PredictedBox | YoloBox) -> float:
    """IoU of two YOLO-normalized center boxes."""
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_predictions(
    predictions: list[PredictedBox],
    ground_truth: list[YoloBox],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy per-class IoU matching for TP / FP / FN inspection."""
    used_gt: set[int] = set()
    true_positives: list[tuple[PredictedBox, YoloBox]] = []
    false_positives: list[PredictedBox] = []

    ordered = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    for pred in ordered:
        best_iou = 0.0
        best_index = -1
        for index, truth in enumerate(ground_truth):
            if index in used_gt or truth.class_id != pred.class_id:
                continue
            iou = yolo_iou(pred, truth)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_index)
            true_positives.append((pred, ground_truth[best_index]))
        else:
            false_positives.append(pred)

    false_negatives = [
        truth for index, truth in enumerate(ground_truth) if index not in used_gt
    ]
    return MatchResult(true_positives, false_positives, false_negatives)


def _xyxy(box: PredictedBox | YoloBox) -> tuple[float, float, float, float]:
    x1 = box.x_center - box.width / 2.0
    y1 = box.y_center - box.height / 2.0
    x2 = box.x_center + box.width / 2.0
    y2 = box.y_center + box.height / 2.0
    return x1, y1, x2, y2
