"""Pothole importer for a *named* YOLO dataset only.

Refuses to run until the manifest lists a real URL, license, and dataset name.
"""

from __future__ import annotations

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
from src.dataset.importers.staging import IMAGE_SUFFIXES, infer_sequence_id
from src.dataset.yolo_labels import parse_yolo_label_file


class PotholeImporter(BaseImporter):
    source_name = "pothole"

    def inspect(self) -> SourceInspectReport:
        raw = self.source.raw_path
        present = raw_has_payload(raw)
        labels = list(raw.rglob("*.txt")) if raw.is_dir() else []
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
            detected_format="yolo" if present else "",
            notes=(
                "Exact pothole dataset not selected. Fill the manifest before import."
                if not self._selection_complete()
                else "Selection fields present; still not APPROVED until reviewed."
            ),
        )

    def parse_objects(self) -> list[ParsedObject]:
        if not self._selection_complete():
            raise ImportError_(
                "Pothole source is not fully documented. Set official_url, license, "
                "and dataset identity in data/sources/datasets.yaml. "
                "Do not import an arbitrary Roboflow project."
            )
        raw = self.source.raw_path
        names = _load_yolo_names(raw)
        pothole_ids = {
            index for index, name in names.items() if name.strip().lower() == "pothole"
        }
        if not pothole_ids:
            raise ImportError_(
                "No class named 'pothole' in the selected dataset. Refusing to guess."
            )
        objects: list[ParsedObject] = []
        for image_path in sorted(
            path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        ):
            label_path = _label_for_image(image_path)
            if label_path is None or not label_path.is_file():
                continue
            width, height = _image_size(image_path)
            split = _split_from_path(image_path)
            for box in parse_yolo_label_file(label_path, max_class_id=max(names)):
                if box.class_id not in pothole_ids:
                    continue
                yolo = YoloBox(box.class_id, box.x_center, box.y_center, box.width, box.height)
                objects.append(
                    ParsedObject(
                        image_path=image_path,
                        source_image=str(_rel(image_path, raw)),
                        source_split=split,
                        sequence_id=infer_sequence_id(image_path),
                        original_class="pothole",
                        box=yolo_to_xyxy(yolo, width, height),
                        image_width=width,
                        image_height=height,
                        conversion_method="yolo_bbox",
                    )
                )
        return objects

    def _selection_complete(self) -> bool:
        return bool(self.source.official_url.strip() and self.source.license.strip())


def _load_yolo_names(raw: Path) -> dict[int, str]:
    for yaml_path in [raw / "data.yaml", raw / "dataset.yaml", *raw.glob("*.yaml")]:
        if not yaml_path.is_file():
            continue
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
    return {0: "pothole"}


def _label_for_image(image_path: Path) -> Path | None:
    sibling = image_path.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    for parent in image_path.parents:
        if parent.name == "images":
            relative = image_path.relative_to(parent).with_suffix(".txt")
            alt = parent.parent / "labels" / relative
            if alt.is_file():
                return alt
    matches = list(image_path.parents[0].rglob(f"{image_path.stem}.txt"))
    return matches[0] if matches else None


def _split_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
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
