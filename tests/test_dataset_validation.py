"""YOLO label parsing and dataset validator tests (no camera, no training)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset.taxonomy import CUSTOM_CLASSES
from src.dataset.validator import validate_dataset
from src.dataset.yolo_labels import LabelError, parse_yolo_line
from src.training.trainer import TrainingNotReadyError, ensure_dataset_ready
from src.utils.paths import project_root


def _write_dataset_yaml(root: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CUSTOM_CLASSES))
    (root / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "taxonomy_version: v0.2\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_valid_yolo_line() -> None:
    box = parse_yolo_line("3 0.5 0.5 0.2 0.4", line_number=1, max_class_id=11)
    assert box.class_id == 3
    assert box.width == pytest.approx(0.2)


def test_malformed_yolo_line() -> None:
    with pytest.raises(LabelError, match="expected 5 fields"):
        parse_yolo_line("3 0.5 0.5", line_number=2, max_class_id=11)


def test_unknown_class_id_in_label() -> None:
    with pytest.raises(LabelError, match="Unknown class id"):
        parse_yolo_line("99 0.5 0.5 0.2 0.2", line_number=1, max_class_id=11)


def test_coordinates_must_be_normalized() -> None:
    with pytest.raises(LabelError, match="outside"):
        parse_yolo_line("0 1.5 0.5 0.2 0.2", line_number=1, max_class_id=11)
    with pytest.raises(LabelError, match="must be > 0"):
        parse_yolo_line("0 0.5 0.5 0.0 0.2", line_number=1, max_class_id=11)


def test_empty_tmp_dataset_is_empty_not_pass(tmp_path: Path) -> None:
    _write_dataset_yaml(tmp_path)
    report = validate_dataset(tmp_path)
    assert report.empty
    assert "CUSTOM DATASET REQUIRED" in report.summary()
    assert "RESULT: EMPTY" in report.summary()


def test_repo_dataset_yaml_is_taxonomy_v02() -> None:
    from src.dataset.taxonomy import assert_dataset_yaml_matches_taxonomy

    assert_dataset_yaml_matches_taxonomy(
        project_root() / "data" / "li_stick_navigation" / "dataset.yaml"
    )


def test_validator_accepts_tiny_labeled_set(tmp_path: Path) -> None:
    _write_dataset_yaml(tmp_path)
    _write_png(tmp_path / "images" / "train" / "a.png")
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    report = validate_dataset(tmp_path)
    assert report.images_total == 1
    assert report.annotations_total == 1
    assert report.passed


def test_validator_flags_missing_image(tmp_path: Path) -> None:
    _write_dataset_yaml(tmp_path)
    (tmp_path / "labels" / "train" / "ghost.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    report = validate_dataset(tmp_path)
    assert report.passed is False
    assert any("Missing image" in issue.message for issue in report.issues)


def test_training_refuses_empty_dataset(tmp_path: Path) -> None:
    _write_dataset_yaml(tmp_path)
    with pytest.raises(TrainingNotReadyError, match="CUSTOM DATASET REQUIRED"):
        ensure_dataset_ready(tmp_path / "dataset.yaml")
