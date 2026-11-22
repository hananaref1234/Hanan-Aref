"""Accessibility Barriers (Zenodo 6382090) CVAT importer.

Taxonomy v0.2 maps `stair` → `stairs` and curb-type `step` → `curb`.
Direction is not invented and is not required for detector training.
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
from src.dataset.importers.cvat import parse_cvat_xml
from src.dataset.importers.staging import IMAGE_SUFFIXES, infer_sequence_id


class AccessibilityBarriersImporter(BaseImporter):
    source_name = "accessibility_barriers"

    def inspect(self) -> SourceInspectReport:
        raw = self.source.raw_path
        present = raw_has_payload(raw)
        xml_files = list(raw.rglob("*.xml")) if raw.is_dir() else []
        images = (
            [path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
            if raw.is_dir()
            else []
        )
        return SourceInspectReport(
            source_name=self.source.name,
            raw_dir=raw,
            present=present,
            annotation_files=len(xml_files),
            image_files=len(images),
            detected_format="cvat_xml" if xml_files else "",
            notes="CC-BY-4.0 zip is 8.0 GB. Taxonomy v0.2 maps stair/curb semantically.",
        )

    def parse_objects(self) -> list[ParsedObject]:
        raw = self.source.raw_path
        objects: list[ParsedObject] = []
        for xml_path in sorted(raw.rglob("*.xml")):
            for image in parse_cvat_xml(xml_path):
                image_path = _resolve_named_image(raw, image.name)
                for box in image.boxes:
                    objects.append(
                        ParsedObject(
                            image_path=image_path,
                            source_image=image.name,
                            source_split="",
                            sequence_id=infer_sequence_id(image_path, fallback=Path(image.name).stem),
                            original_class=box.label,
                            box=PixelBox(box.xtl, box.ytl, box.xbr, box.ybr),
                            image_width=image.width,
                            image_height=image.height,
                            attributes=dict(box.attributes),
                            conversion_method="bbox",
                            original_annotation=(
                                f"cvat label={box.label} attrs={box.attributes}"
                            ),
                        )
                    )
        return objects


def _resolve_named_image(raw: Path, name: str) -> Path:
    direct = raw / name
    if direct.exists():
        return direct
    matches = list(raw.rglob(Path(name).name))
    if matches:
        return matches[0]
    return direct
