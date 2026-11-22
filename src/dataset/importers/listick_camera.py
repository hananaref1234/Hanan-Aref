"""First-party cane-camera importer (YOLO labels already in Li-Stick names)."""

from __future__ import annotations

from pathlib import Path

from src.dataset.importers.base_importer import (
    BaseImporter,
    ParsedObject,
    SourceInspectReport,
    raw_has_payload,
)
from src.dataset.importers.boxes import YoloBox, yolo_to_xyxy
from src.dataset.importers.staging import IMAGE_SUFFIXES, infer_sequence_id
from src.dataset.taxonomy import CUSTOM_CLASSES
from src.dataset.yolo_labels import parse_yolo_label_file


class ListickCameraImporter(BaseImporter):
    source_name = "listick_camera"

    def inspect(self) -> SourceInspectReport:
        raw = self.source.raw_path
        present = raw_has_payload(raw)
        images = (
            [path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
            if raw.is_dir()
            else []
        )
        return SourceInspectReport(
            source_name=self.source.name,
            raw_dir=raw,
            present=present,
            image_files=len(images),
            detected_format="yolo" if present else "",
            notes="First-party Camera Module V2 NoIR captures. Empty until collection.",
        )

    def parse_objects(self) -> list[ParsedObject]:
        raw = self.source.raw_path
        objects: list[ParsedObject] = []
        max_id = len(CUSTOM_CLASSES) - 1
        for image_path in sorted(
            path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        ):
            label_path = image_path.with_suffix(".txt")
            if not label_path.is_file():
                alt = image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
                label_path = alt if alt.is_file() else label_path
            if not label_path.is_file():
                continue
            import cv2

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            split = ""
            parts = {part.lower() for part in image_path.parts}
            if "train" in parts:
                split = "train"
            elif "val" in parts:
                split = "val"
            elif "test" in parts:
                split = "test"
            for box in parse_yolo_label_file(label_path, max_class_id=max_id):
                yolo = YoloBox(box.class_id, box.x_center, box.y_center, box.width, box.height)
                objects.append(
                    ParsedObject(
                        image_path=image_path,
                        source_image=image_path.name,
                        source_split=split,
                        sequence_id=infer_sequence_id(image_path),
                        original_class=CUSTOM_CLASSES[box.class_id],
                        box=yolo_to_xyxy(yolo, width, height),
                        image_width=int(width),
                        image_height=int(height),
                        conversion_method="yolo_bbox",
                    )
                )
        return objects
