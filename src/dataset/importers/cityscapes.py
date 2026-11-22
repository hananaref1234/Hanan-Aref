"""Cityscapes semantic-mask importer.

wall and pole boxes are derived from connected components. vegetation is never
mapped to tree_trunk. Automatic download is disabled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.dataset.importers.base_importer import (
    BaseImporter,
    ParsedObject,
    SourceInspectReport,
    raw_has_payload,
)
from src.dataset.importers.mask_to_box import MASK_TO_BOX_LOSS, boxes_from_label_map
from src.dataset.importers.staging import infer_sequence_id

# Official Cityscapes label IDs (not trainIds).
CITYSCAPES_ID_TO_NAME = {
    12: "wall",
    17: "pole",
    21: "vegetation",
    24: "person",
    26: "car",
    32: "motorcycle",
    33: "bicycle",
}

IMPORT_CLASS_IDS = {12, 17, 24, 26, 32, 33}


class CityscapesImporter(BaseImporter):
    source_name = "cityscapes"

    def inspect(self) -> SourceInspectReport:
        raw = self.source.raw_path
        present = raw_has_payload(raw)
        labels = list(raw.rglob("*_gtFine_labelIds.png")) if raw.is_dir() else []
        images = list(raw.rglob("*_leftImg8bit.png")) if raw.is_dir() else []
        return SourceInspectReport(
            source_name=self.source.name,
            raw_dir=raw,
            present=present,
            annotation_files=len(labels),
            image_files=len(images),
            detected_format="cityscapes_labelids" if labels else "",
            notes=MASK_TO_BOX_LOSS,
        )

    def parse_objects(self) -> list[ParsedObject]:
        import cv2

        raw = self.source.raw_path
        objects: list[ParsedObject] = []
        for label_path in sorted(raw.rglob("*_gtFine_labelIds.png")):
            image_path = _rgb_for_label(label_path, raw)
            label_map = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if label_map is None:
                continue
            if label_map.ndim == 3:
                label_map = label_map[:, :, 0]
            height, width = label_map.shape[:2]
            split = _split_from_path(label_path)
            sequence_id = _cityscapes_sequence(label_path)
            for mask_box in boxes_from_label_map(
                np.asarray(label_map),
                IMPORT_CLASS_IDS,
                min_pixels=32,
            ):
                class_name = CITYSCAPES_ID_TO_NAME.get(mask_box.class_id)
                if class_name is None:
                    continue
                objects.append(
                    ParsedObject(
                        image_path=image_path,
                        source_image=str(_rel(image_path, raw)),
                        source_split=split,
                        sequence_id=sequence_id or infer_sequence_id(image_path),
                        original_class=class_name,
                        box=mask_box.box,
                        image_width=int(width),
                        image_height=int(height),
                        conversion_method="mask_to_box",
                        original_annotation=(
                            f"labelIds={mask_box.class_id} pixels={mask_box.pixel_count} "
                            f"{MASK_TO_BOX_LOSS}"
                        ),
                    )
                )
        return objects


def _rgb_for_label(label_path: Path, raw: Path) -> Path:
    name = label_path.name.replace("_gtFine_labelIds.png", "_leftImg8bit.png")
    matches = list(raw.rglob(name))
    if matches:
        return matches[0]
    return label_path.with_name(name)


def _split_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if "val" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return ""


def _cityscapes_sequence(path: Path) -> str:
    # frankfurt_000000_000294_gtFine_labelIds.png -> frankfurt_000000
    stem = path.name.replace("_gtFine_labelIds.png", "").replace("_leftImg8bit.png", "")
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def _rel(path: Path, raw: Path) -> Path:
    try:
        return path.relative_to(raw)
    except ValueError:
        return Path(path.name)
