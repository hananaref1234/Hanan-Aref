"""TimberVision research adapter (download blocked until license review)."""

from __future__ import annotations

from src.dataset.importers.base_importer import (
    BaseImporter,
    ImportError_,
    ParsedObject,
    SourceInspectReport,
    raw_has_payload,
)


class TimberVisionImporter(BaseImporter):
    source_name = "timbervision"

    def inspect(self) -> SourceInspectReport:
        raw = self.source.raw_path
        present = raw_has_payload(raw)
        images = 0
        labels = 0
        if present and raw.is_dir():
            images = sum(
                1 for path in raw.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            labels = sum(1 for path in raw.rglob("*.txt") if "labels" in path.parts)
        return SourceInspectReport(
            source_name=self.source.name,
            raw_dir=raw,
            present=present,
            annotation_files=labels,
            image_files=images,
            detected_format="timbervision_yolo_seg" if present else "",
            notes=(
                "TimberVision is CC BY-NC-SA 4.0 and forestry-domain. "
                "License review required before import."
            ),
        )

    def parse_objects(self) -> list[ParsedObject]:
        raise ImportError_(
            "TimberVision import is blocked until LICENSE_REVIEW_REQUIRED is cleared. "
            "License is CC BY-NC-SA 4.0 (non-commercial, share-alike). "
            "Annotations target forestry logs/trunks, not sidewalk tree_trunk."
        )
