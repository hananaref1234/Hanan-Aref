"""Staging provenance CSV: every imported box stays traceable."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path

PROVENANCE_FIELDS = (
    "sample_id",
    "source_dataset",
    "source_image",
    "source_split",
    "original_class",
    "mapped_class",
    "mapping_type",
    "review_status",
    "sequence_id",
    "license",
    "proposed_class",
    "taxonomy",
    "conversion_method",
    "original_annotation",
    "staging_image",
    "staging_label",
    "notes",
    "taxonomy_version",
    "source_subtype",
    "direction_status",
    "direction_class",
    "detector_split",
    "review_method",
)

ALLOWED_MAPPING_TYPES = (
    "DIRECT",
    "MANUAL_DIRECTION",
    "SEMANTIC_REVIEW",
    "MASK_TO_BOX",
    "CUSTOM",
)


@dataclass
class ProvenanceRecord:
    sample_id: str
    source_dataset: str
    source_image: str
    source_split: str
    original_class: str
    mapped_class: str
    mapping_type: str
    review_status: str
    sequence_id: str
    license: str
    proposed_class: str = ""
    taxonomy: str = "custom"
    conversion_method: str = ""
    original_annotation: str = ""
    staging_image: str = ""
    staging_label: str = ""
    notes: str = ""
    taxonomy_version: str = ""
    source_subtype: str = ""
    direction_status: str = ""
    direction_class: str = ""
    detector_split: str = ""
    review_method: str = ""

    def to_row(self) -> dict[str, str]:
        return {field: str(getattr(self, field) or "") for field in PROVENANCE_FIELDS}


def default_metadata_path(staging_root: Path) -> Path:
    return staging_root / "staging_metadata.csv"


def write_provenance(path: Path, records: list[ProvenanceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROVENANCE_FIELDS)
        writer.writeheader()
        for record in records:
            if record.mapping_type not in ALLOWED_MAPPING_TYPES:
                raise ValueError(f"Invalid mapping_type {record.mapping_type!r}.")
            writer.writerow(record.to_row())


def append_provenance(path: Path, records: list[ProvenanceRecord]) -> None:
    existing = read_provenance(path) if path.is_file() else []
    write_provenance(path, existing + records)


def read_provenance(path: Path) -> list[ProvenanceRecord]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        missing = [
            field
            for field in ("sample_id", "source_dataset")
            if field not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(missing)}.")
        field_names = [item.name for item in fields(ProvenanceRecord)]
        records: list[ProvenanceRecord] = []
        for raw in reader:
            kwargs = {name: str(raw.get(name, "") or "") for name in field_names}
            records.append(ProvenanceRecord(**kwargs))
        return records


def make_sample_id(
    source_dataset: str,
    source_split: str,
    image_stem: str,
    box_index: int,
) -> str:
    split = source_split or "unsplit"
    return f"{source_dataset}_{split}_{image_stem}_{box_index:04d}"
