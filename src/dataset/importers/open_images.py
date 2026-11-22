"""Open Images V7 research adapter.

Open Images does not provide Li-Stick Exp2 target classes (pole, bollard,
barrier, tree_trunk). Broad classes such as ``Tree`` and ``Street light``
must not be auto-mapped. This importer documents the source and refuses
conversion until a future, explicitly approved subset is configured.
"""

from __future__ import annotations

from src.dataset.importers.base_importer import (
    BaseImporter,
    ImportError_,
    ParsedObject,
    SourceInspectReport,
    raw_has_payload,
)


class OpenImagesImporter(BaseImporter):
    source_name = "open_images_v7"

    def inspect(self) -> SourceInspectReport:
        raw = self.source.raw_path
        present = raw_has_payload(raw)
        return SourceInspectReport(
            source_name=self.source.name,
            raw_dir=raw,
            present=present,
            annotation_files=0,
            image_files=0,
            detected_format="open_images_v7" if present else "",
            notes=(
                "Open Images has no pole/bollard/barrier/tree_trunk bbox classes. "
                "Do not map Tree or Street light automatically."
            ),
        )

    def parse_objects(self) -> list[ParsedObject]:
        raise ImportError_(
            "Open Images V7 is recorded for research only. It does not expose "
            "Li-Stick Exp2 classes (pole, bollard, barrier, tree_trunk). "
            "Broad classes Tree and Street light are rejected for automatic mapping. "
            "Use Roboflow or other vetted sidewalk sources instead."
        )
