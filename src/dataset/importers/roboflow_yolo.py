"""Roboflow Universe YOLO export importer.

Only sources with documented licenses and MANUAL_REVIEW_REQUIRED (or APPROVED)
status may be imported. Downloads are manual via Roboflow Universe unless the
user supplies a Roboflow API key separately.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from src.dataset.importers.base_importer import (
    BaseImporter,
    ImportError_,
    ParsedObject,
    SourceInspectReport,
    raw_has_payload,
)
from src.dataset.importers.boxes import YoloBox, yolo_to_xyxy
from src.dataset.importers.sampling import cap_instances_by_image
from src.dataset.importers.staging import IMAGE_SUFFIXES, infer_sequence_id
from src.dataset.yolo_labels import parse_yolo_annotation_file


class RoboflowYoloImporter(BaseImporter):
    """Import a Roboflow-exported YOLO dataset placed under ``data/raw/<name>/``."""

    source_name = "roboflow_yolo"

    def inspect(self) -> SourceInspectReport:
        raw = self.source.raw_path
        present = raw_has_payload(raw)
        labels = [
            path
            for path in raw.rglob("*.txt")
            if path.name.lower() not in {"readme.txt", "train.txt", "val.txt", "test.txt"}
            and "labels" in {part.lower() for part in path.parts}
        ] if raw.is_dir() else []
        if not labels:
            labels = [
                path
                for path in raw.rglob("*.txt")
                if path.suffix.lower() == ".txt"
                and path.name.lower() not in {"readme.txt", "train.txt", "val.txt", "test.txt"}
                and path.with_suffix(".jpg").exists() is False
                and any(path.with_suffix(ext).exists() for ext in (".jpg", ".jpeg", ".png"))
            ] if raw.is_dir() else []
        images = (
            [path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
            if raw.is_dir()
            else []
        )
        return SourceInspectReport(
            source_name=self.source.name,
            raw_dir=raw,
            present=present,
            annotation_files=len(labels),
            image_files=len(images),
            detected_format="roboflow_yolo" if present else "",
            notes=(
                "Roboflow YOLO export not present. Download from the official "
                "Universe page into this raw directory."
                if not present
                else "Roboflow YOLO export detected."
            ),
        )

    def parse_objects(self) -> list[ParsedObject]:
        raw = self.source.raw_path
        names = _load_yolo_names(raw)
        if not names:
            raise ImportError_(f"No YOLO class names found under {raw}.")

        allowed = _allowed_source_classes(self.source, names)
        if not allowed:
            raise ImportError_(
                f"No mapped classes configured for {self.source.name}. "
                "Check listick_mapping in data/sources/datasets.yaml."
            )

        max_instances = _extra_int(self.source, "max_instances_per_class", default=2000)
        seed = _extra_int(self.source, "sample_seed", default=42)

        by_class_image: dict[str, dict[str, list[ParsedObject]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for image_path in sorted(
            path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        ):
            label_path = _label_for_image(image_path, raw)
            if label_path is None or not label_path.is_file():
                continue
            width, height = _image_size(image_path)
            split = _split_from_path(image_path, raw)
            rel = _rel(image_path, raw)
            image_key = str(rel)
            label_rel = _rel(label_path, raw)
            for annotation in parse_yolo_annotation_file(label_path, max_class_id=max(names)):
                if annotation.class_id not in names:
                    continue
                original = names[annotation.class_id].strip()
                if not _class_allowed(self.source, original):
                    continue
                class_key = _normalize_roboflow_class(original) or original
                yolo = YoloBox(
                    annotation.class_id,
                    annotation.x_center,
                    annotation.y_center,
                    annotation.width,
                    annotation.height,
                )
                conversion_method = (
                    "POLYGON_TO_BOX" if annotation.source_format == "POLYGON" else "YOLO_BBOX"
                )
                parsed = ParsedObject(
                    image_path=image_path,
                    source_image=str(rel),
                    source_split=split,
                    sequence_id=infer_sequence_id(image_path),
                    original_class=original,
                    box=yolo_to_xyxy(yolo, width, height),
                    image_width=width,
                    image_height=height,
                    conversion_method=conversion_method,
                    original_annotation=_original_annotation(
                        annotation=annotation,
                        label_rel=label_rel,
                    ),
                )
                by_class_image[class_key][image_key].append(parsed)

        objects: list[ParsedObject] = []
        for class_name in sorted(by_class_image):
            grouped = by_class_image[class_name]
            objects.extend(
                cap_instances_by_image(grouped, max_instances=max_instances, seed=seed)
            )
        return objects


def _normalize_roboflow_class(name: str) -> str:
    cleaned = name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def _allowed_source_classes(source, names: dict[int, str]) -> set[str]:
    keys = set(source.listick_mapping)
    if not keys:
        return set()
    return keys


def _class_allowed(source, class_name: str) -> bool:
    norm = _normalize_roboflow_class(class_name)
    extra_only = source.extra or {}
    only = extra_only.get("import_classes")
    if isinstance(only, list) and only:
        only_norm = {_normalize_roboflow_class(str(item)) for item in only}
        if norm not in only_norm and class_name not in only:
            return False
    for key in source.listick_mapping:
        if class_name == key or norm == _normalize_roboflow_class(key):
            return True
    return False


def _extra_int(source, key: str, *, default: int) -> int:
    extra = source.extra or {}
    value = extra.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_yolo_names(raw: Path) -> dict[int, str]:
    for yaml_path in [raw / "data.yaml", raw / "dataset.yaml", *sorted(raw.glob("*.yaml"))]:
        if not yaml_path.is_file():
            continue
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
    return {}


def _label_for_image(image_path: Path, raw: Path) -> Path | None:
    sibling = image_path.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    for parent in image_path.parents:
        if parent.name.lower() == "images":
            relative = image_path.relative_to(parent).with_suffix(".txt")
            alt = parent.parent / "labels" / relative
            if alt.is_file():
                return alt
        if parent == raw:
            break
    matches = [
        path
        for path in image_path.parent.rglob(f"{image_path.stem}.txt")
        if "labels" in {part.lower() for part in path.parts}
    ]
    return matches[0] if matches else None


def _split_from_path(path: Path, raw: Path) -> str:
    try:
        rel_parts = path.relative_to(raw).parts
    except ValueError:
        rel_parts = path.parts
    parts = {part.lower() for part in rel_parts}
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return ""


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return int(width), int(height)


def _rel(path: Path, raw: Path) -> Path:
    try:
        return path.relative_to(raw)
    except ValueError:
        return Path(path.name)


def _original_annotation(annotation, label_rel: Path) -> str:
    if annotation.source_format == "POLYGON":
        points = ";".join(f"{x:.6f},{y:.6f}" for x, y in annotation.polygon_points)
        return (
            f"source_label={label_rel};format=polygon;line={annotation.line_number};"
            f"points={len(annotation.polygon_points)};polygon={points}"
        )
    return (
        f"source_label={label_rel};format=bbox;line={annotation.line_number};"
        f"xywh={annotation.x_center:.6f},{annotation.y_center:.6f},"
        f"{annotation.width:.6f},{annotation.height:.6f}"
    )
