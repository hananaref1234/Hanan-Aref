"""DoorDetect importer.

Open doors are proposed as open_doorway but never auto-approved.
Semi-open doors are not mapped to a Li-Stick class.
"""

from __future__ import annotations

from pathlib import Path

from src.dataset.importers.base_importer import (
    BaseImporter,
    ParsedObject,
    SourceInspectReport,
    raw_has_payload,
)
from src.dataset.importers.boxes import PixelBox
from src.dataset.importers.staging import IMAGE_SUFFIXES, infer_sequence_id

STATE_FOLDERS = {
    "open": "open",
    "closed": "closed",
    "semi-open": "semi-open",
    "semi_open": "semi-open",
    "semiopen": "semi-open",
}

SPLIT_FOLDERS = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
}


class DoorDetectImporter(BaseImporter):
    source_name = "doordetect"

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
            annotation_files=0,
            image_files=len(images),
            detected_format="folder_classes" if present else "",
            notes=(
                "License unspecified on GitHub. Open ≠ traversable doorway. "
                "Prefer OriginalSize RGB splits over tightly cropped door patches."
            ),
        )

    def parse_objects(self) -> list[ParsedObject]:
        raw = self.source.raw_path
        objects: list[ParsedObject] = []
        for image_path in sorted(
            path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        ):
            state = _state_from_path(image_path)
            if state is None:
                continue
            if _looks_like_depth(image_path):
                continue
            width, height = _image_size(image_path)
            split = _split_from_path(image_path)
            # Full-frame box: classification images are already door-centric.
            # Reviewer must still decide whether the opening is a traversable doorway.
            objects.append(
                ParsedObject(
                    image_path=image_path,
                    source_image=str(image_path.relative_to(raw)),
                    source_split=split,
                    sequence_id=infer_sequence_id(image_path),
                    original_class=state,
                    box=PixelBox(0.0, 0.0, float(width), float(height)),
                    image_width=width,
                    image_height=height,
                    conversion_method="image_class_full_frame",
                    original_annotation=f"folder_class={state}",
                )
            )
        return objects


def _state_from_path(path: Path) -> str | None:
    for part in path.parts:
        key = part.lower().replace(" ", "-")
        if key in STATE_FOLDERS:
            return STATE_FOLDERS[key]
    return None


def _split_from_path(path: Path) -> str:
    for part in path.parts:
        key = part.lower()
        if key in SPLIT_FOLDERS:
            return SPLIT_FOLDERS[key]
    return ""


def _looks_like_depth(path: Path) -> bool:
    return "depth" in {part.lower() for part in path.parts}


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return 640, 480
    height, width = image.shape[:2]
    return int(width), int(height)
