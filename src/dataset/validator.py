"""Dataset integrity checks for the Li-Stick navigation YOLO dataset."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.dataset.leakage import LeakageIssue, find_split_leakage, parse_metadata_csv
from src.dataset.taxonomy import (
    COMBINED_CLASSES,
    CUSTOM_CLASSES,
    TAXONOMY_VERSION,
    load_dataset_yaml,
    load_dataset_yaml_names,
)
from src.dataset.yolo_labels import LabelError, parse_yolo_label_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass
class ValidationIssue:
    severity: str  # error | warning
    message: str


@dataclass
class DatasetValidationReport:
    dataset_root: Path
    images_total: int = 0
    annotations_total: int = 0
    split_image_counts: dict[str, int] = field(default_factory=dict)
    empty_label_files: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def passed(self) -> bool:
        return self.images_total > 0 and self.error_count == 0

    @property
    def empty(self) -> bool:
        return self.images_total == 0

    def summary(self) -> str:
        invalid_labels = sum(1 for issue in self.issues if "Malformed" in issue.message or "label" in issue.message.lower() and issue.severity == "error")
        missing_images = sum(1 for issue in self.issues if "Missing image" in issue.message)
        unknown_classes = sum(1 for issue in self.issues if "Unknown class" in issue.message)
        lines = [
            "Dataset Validation",
            "------------------",
            f"Images: {self.images_total}",
            f"Annotations: {self.annotations_total}",
            "",
            f"Invalid labels: {invalid_labels}",
            f"Missing images: {missing_images}",
            f"Unknown classes: {unknown_classes}",
            f"Empty label files: {self.empty_label_files}",
            "",
            f"Train: {self.split_image_counts.get('train', 0)}",
            f"Val: {self.split_image_counts.get('val', 0)}",
            f"Test: {self.split_image_counts.get('test', 0)}",
            "",
        ]
        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.message}")
            lines.append("")
        if self.empty:
            lines.append("RESULT: EMPTY")
            lines.append("CUSTOM DATASET REQUIRED")
        elif self.passed:
            lines.append("RESULT: PASS")
        else:
            lines.append("RESULT: FAIL")
        return "\n".join(lines)


def validate_dataset(dataset_root: Path) -> DatasetValidationReport:
    """Validate images, labels, class IDs, and optional split-leakage metadata."""
    dataset_root = dataset_root.resolve()
    report = DatasetValidationReport(dataset_root=dataset_root)
    yaml_path = dataset_root / "dataset.yaml"
    if not yaml_path.exists():
        report.issues.append(
            ValidationIssue("error", f"dataset.yaml not found at {yaml_path}.")
        )
        return report

    try:
        data = load_dataset_yaml(yaml_path)
        names = load_dataset_yaml_names(yaml_path)
    except ValueError as exc:
        report.issues.append(ValidationIssue("error", str(exc)))
        return report

    version = str(data.get("taxonomy_version", "")).strip()
    if version != TAXONOMY_VERSION:
        report.issues.append(
            ValidationIssue(
                "error",
                f"dataset.yaml taxonomy_version must be {TAXONOMY_VERSION!r}, "
                f"found {version!r}.",
            )
        )

    training_view = str(data.get("training_view", "")).strip()
    if training_view:
        max_class_id = _validate_training_view_names(names, report)
    else:
        expected = dict(enumerate(CUSTOM_CLASSES))
        if names != expected:
            report.issues.append(
                ValidationIssue(
                    "error",
                    "dataset.yaml class mapping does not match the frozen custom taxonomy.",
                )
            )
        max_class_id = max(expected) if expected else -1

    all_image_names: dict[str, list[str]] = defaultdict(list)
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = _list_images(image_dir) if image_dir.exists() else []
        labels = _list_labels(label_dir) if label_dir.exists() else []
        report.split_image_counts[split] = len(images)
        report.images_total += len(images)

        image_stems = {path.stem: path for path in images}
        label_stems = {path.stem: path for path in labels}

        for path in images:
            all_image_names[path.name].append(split)
            label_path = label_stems.get(path.stem)
            if label_path is None:
                report.issues.append(
                    ValidationIssue(
                        "error",
                        f"Missing label for image {path.relative_to(dataset_root)}. "
                        "Hard negatives still need an empty .txt file.",
                    )
                )
                continue
            _validate_image(path, report)
            boxes = _validate_label(label_path, max_class_id, report)
            if boxes == [] and label_path.exists() and not label_path.read_text(encoding="utf-8").strip():
                report.empty_label_files += 1
                report.issues.append(
                    ValidationIssue(
                        "warning",
                        f"Empty label file {label_path.relative_to(dataset_root)} "
                        "(valid if this is a hard-negative image).",
                    )
                )
            report.annotations_total += len(boxes)

        for stem, label_path in label_stems.items():
            if stem not in image_stems:
                report.issues.append(
                    ValidationIssue(
                        "error",
                        f"Missing image for label {label_path.relative_to(dataset_root)}.",
                    )
                )

    for name, splits in all_image_names.items():
        if len(splits) > 1:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"Duplicate filename {name!r} appears in splits: {', '.join(splits)}.",
                )
            )

    metadata_path = dataset_root / "metadata.csv"
    if metadata_path.exists():
        _validate_metadata(metadata_path, report)
    else:
        report.issues.append(
            ValidationIssue(
                "warning",
                "No metadata.csv found. Split leakage by video/session/sequence "
                "cannot be checked. Add metadata before using walking-video frames.",
            )
        )

    return report


def _validate_training_view_names(
    names: dict[int, str],
    report: DatasetValidationReport,
) -> int:
    """Allow a contiguous subset of CUSTOM_CLASSES for derived training views."""
    if not names:
        report.issues.append(
            ValidationIssue("error", "training view dataset.yaml has an empty names mapping.")
        )
        return -1
    ids = sorted(names)
    if ids != list(range(len(ids))):
        report.issues.append(
            ValidationIssue(
                "error",
                "training view class IDs must be contiguous starting at 0 "
                f"(found {ids}). Ultralytics requires this; do not use sparse global IDs.",
            )
        )
    seen: set[str] = set()
    for class_id, name in sorted(names.items()):
        if name not in COMBINED_CLASSES:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"training view class {name!r} (id {class_id}) is not in taxonomy "
                    f"{TAXONOMY_VERSION} custom or combined-generic classes.",
                )
            )
        if name in seen:
            report.issues.append(
                ValidationIssue("error", f"training view repeats class name {name!r}.")
            )
        seen.add(name)
    return max(names) if names else -1


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _list_labels(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )


def _validate_image(path: Path, report: DatasetValidationReport) -> None:
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        report.issues.append(
            ValidationIssue("error", f"Corrupt or unreadable image {path}.")
        )


def _validate_label(
    path: Path,
    max_class_id: int,
    report: DatasetValidationReport,
) -> list:
    try:
        return parse_yolo_label_file(path, max_class_id=max_class_id)
    except LabelError as exc:
        report.issues.append(ValidationIssue("error", str(exc)))
        return []


def _validate_metadata(path: Path, report: DatasetValidationReport) -> None:
    try:
        rows = parse_metadata_csv(path)
    except ValueError as exc:
        report.issues.append(ValidationIssue("error", f"Invalid metadata.csv: {exc}"))
        return
    leaks = find_split_leakage(rows)
    for leak in leaks:
        report.issues.append(
            ValidationIssue(
                "error",
                _format_leak(leak),
            )
        )


def _format_leak(leak: LeakageIssue) -> str:
    return (
        f"Split leakage: {leak.group_field}={leak.group_value!r} appears in "
        f"{', '.join(leak.splits)} ({len(leak.filenames)} files). "
        "Keep an entire video/session/sequence in one split."
    )
