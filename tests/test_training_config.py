"""Training YAML validation tests (no GPU, no training run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.config import TrainingConfigError, load_training_config
from src.training.matching import PredictedBox, match_predictions, yolo_iou
from src.dataset.yolo_labels import YoloBox
from src.utils.paths import project_root


def test_load_default_training_config() -> None:
    config = load_training_config()
    assert config.base_model == "yolo26n.pt"
    assert config.epochs == 100
    assert config.image_size == 640
    assert config.augmentation.flipud == 0.0
    assert config.dataset_yaml == project_root() / "data" / "li_stick_navigation" / "dataset.yaml"


def test_invalid_epochs(tmp_path: Path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text(
        "model:\n  base_model: yolo26n.pt\ntraining:\n  epochs: 0\n  image_size: 640\n  batch_size: 8\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainingConfigError, match="epochs"):
        load_training_config(path)


def test_vertical_flip_rejected(tmp_path: Path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text(
        "model:\n  base_model: yolo26n.pt\n"
        "training:\n  epochs: 1\n  image_size: 640\n  batch_size: 8\n"
        "augmentation:\n  flipud: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainingConfigError, match="flipud"):
        load_training_config(path)


def test_iou_matching_true_positive() -> None:
    pred = PredictedBox(3, 0.5, 0.5, 0.2, 0.2, 0.9)
    truth = YoloBox(3, 0.51, 0.5, 0.2, 0.2, 1)
    assert yolo_iou(pred, truth) > 0.5
    result = match_predictions([pred], [truth], iou_threshold=0.5)
    assert len(result.true_positives) == 1
    assert result.false_positives == []
    assert result.false_negatives == []


def test_iou_matching_false_positive_and_negative() -> None:
    pred = PredictedBox(0, 0.2, 0.2, 0.1, 0.1, 0.8)
    truth = YoloBox(3, 0.8, 0.8, 0.1, 0.1, 1)
    result = match_predictions([pred], [truth], iou_threshold=0.5)
    assert result.true_positives == []
    assert len(result.false_positives) == 1
    assert len(result.false_negatives) == 1
