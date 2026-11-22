"""Mapillary Vistas is documented only. Import and download are disabled."""

from __future__ import annotations

from src.dataset.importers.base_importer import (
    BaseImporter,
    ImportError_,
    ParsedObject,
    SourceInspectReport,
)
from src.dataset.importers.download import DownloadBlockedError


class MapillaryVistasImporter(BaseImporter):
    source_name = "mapillary_vistas"

    def inspect(self) -> SourceInspectReport:
        return SourceInspectReport(
            source_name=self.source.name,
            raw_dir=self.source.raw_path,
            present=False,
            notes="LICENSE_REVIEW_REQUIRED. Importer disabled. Do not download.",
        )

    def parse_objects(self) -> list[ParsedObject]:
        raise DownloadBlockedError(
            "Mapillary Vistas is LICENSE_REVIEW_REQUIRED. Import is disabled."
        )

    def convert(self, *, include_generic: bool = False) -> None:  # type: ignore[override]
        raise DownloadBlockedError(
            "Mapillary Vistas is LICENSE_REVIEW_REQUIRED. Import is disabled."
        )
