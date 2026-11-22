"""Validate staging provenance without promoting or training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.dataset.importers.boxes import PixelBox
from src.dataset.importers.provenance import PROVENANCE_FIELDS, ProvenanceRecord
from src.dataset.importers.review import REVIEW_STATUSES, class_for_promotion
from src.dataset.taxonomy import CUSTOM_CLASSES, DIRECTIONAL_CLASSES_V0_1, TAXONOMY_VERSION


def validate_staging_records(records: list[ProvenanceRecord]) -> list[str]:
    """Return issue strings. Empty means the staging table is consistent."""
    issues: list[str] = []
    ids = [record.sample_id for record in records]
    dup_ids = [key for key, count in Counter(ids).items() if count > 1]
    if dup_ids:
        issues.append(f"Duplicate sample_id values: {dup_ids[:10]}")
    for record in records:
        prefix = record.sample_id or "<missing sample_id>"
        if not record.sample_id:
            issues.append("Row missing sample_id")
        if not record.source_dataset:
            issues.append(f"{prefix}: missing source_dataset")
        if not record.source_image:
            issues.append(f"{prefix}: missing source_image")
        if not record.original_class:
            issues.append(f"{prefix}: missing original_class")
        if not record.license:
            issues.append(f"{prefix}: missing license")
        if record.mapping_type not in {
            "DIRECT",
            "MANUAL_DIRECTION",
            "SEMANTIC_REVIEW",
            "MASK_TO_BOX",
            "CUSTOM",
        }:
            issues.append(f"{prefix}: invalid mapping_type {record.mapping_type!r}")
        if record.review_status not in REVIEW_STATUSES:
            issues.append(f"{prefix}: invalid review_status {record.review_status!r}")
        if record.mapped_class and record.mapped_class not in CUSTOM_CLASSES:
            issues.append(f"{prefix}: mapped_class {record.mapped_class!r} not in taxonomy")
        if record.direction_class and record.direction_class not in DIRECTIONAL_CLASSES_V0_1:
            issues.append(
                f"{prefix}: direction_class {record.direction_class!r} is not a v0.1 direction label"
            )
        if record.taxonomy_version and record.taxonomy_version != TAXONOMY_VERSION:
            issues.append(
                f"{prefix}: taxonomy_version {record.taxonomy_version!r} != {TAXONOMY_VERSION}"
            )
        if record.review_status == "NEEDS_DIRECTION_REVIEW" and record.mapped_class:
            issues.append(f"{prefix}: directional review row already has mapped_class")
        if class_for_promotion(record) and record.review_status in {
            "PENDING",
            "NEEDS_DIRECTION_REVIEW",
            "UNMAPPED_SOURCE_CLASS",
        }:
            issues.append(f"{prefix}: unreviewed row is marked promotable")
        if "xyxy=" in record.original_annotation:
            try:
                _parse_xyxy(record.original_annotation)
            except ValueError as exc:
                issues.append(f"{prefix}: malformed box ({exc})")
        if record.staging_image:
            path = Path(record.staging_image)
            if record.review_status == "NEEDS_DIRECTION_REVIEW" and not path.is_file():
                issues.append(f"{prefix}: missing staging image {path}")
    _ = PROVENANCE_FIELDS
    return issues


def _parse_xyxy(text: str) -> PixelBox:
    xyxy_part = text.split("xyxy=", 1)[1].split(";", 1)[0]
    x1, y1, x2, y2 = (float(part) for part in xyxy_part.split(","))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty box {x1},{y1},{x2},{y2}")
    return PixelBox(x1, y1, x2, y2)
