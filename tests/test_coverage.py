"""Coverage status for taxonomy v0.2. No training."""

from __future__ import annotations

from src.dataset.importers.coverage import (
    STATUS_GOOD,
    STATUS_NO_DATA,
    coverage_from_records,
    format_coverage_table,
)
from src.dataset.importers.provenance import ProvenanceRecord
from src.dataset.taxonomy import CUSTOM_CLASSES


def _record(**overrides) -> ProvenanceRecord:
    data = {
        "sample_id": "s1",
        "source_dataset": "accessibility_barriers",
        "source_image": "images/a.jpg",
        "source_split": "",
        "original_class": "stair",
        "mapped_class": "stairs",
        "mapping_type": "DIRECT",
        "review_status": "APPROVED",
        "sequence_id": "a",
        "license": "CC-BY-4.0",
        "notes": "Taxonomy v0.2 semantic mapping.",
        "taxonomy_version": "v0.2",
    }
    data.update(overrides)
    return ProvenanceRecord(**data)


def test_empty_coverage_is_no_data() -> None:
    rows = coverage_from_records([])
    assert [row.name for row in rows] == list(CUSTOM_CLASSES)
    assert all(row.instances == 0 and row.status == STATUS_NO_DATA for row in rows)
    table = format_coverage_table(rows)
    assert "IMAGES" in table
    assert "INSTANCES" in table
    assert "drop_off" in table
    assert "stairs_up" not in table


def test_semantic_stairs_and_curb_are_counted() -> None:
    rows = coverage_from_records(
        [
            _record(sample_id="a", source_image="images/a.jpg"),
            _record(
                sample_id="b",
                source_image="images/b.jpg",
                original_class="step",
                mapped_class="curb",
                source_subtype="curb",
            ),
            _record(
                sample_id="c",
                source_image="images/c.jpg",
                original_class="step",
                mapped_class="",
                review_status="UNMAPPED_SOURCE_CLASS",
                notes="doorstep",
            ),
        ]
    )
    by_name = {row.name: row for row in rows}
    assert by_name["stairs"].instances == 1
    assert by_name["stairs"].images == 1
    assert by_name["curb"].instances == 1
    assert by_name["open_doorway"].instances == 0
    assert by_name["stairs"].status == "WEAK"
    assert "accessibility_barriers" in by_name["stairs"].sources


def test_preserved_direction_still_counts_as_semantic_stairs() -> None:
    rows = coverage_from_records(
        [
            _record(
                review_status="STAIRS_DOWN",
                mapped_class="stairs",
                direction_status="STAIRS_DOWN",
                direction_class="stairs_down",
            )
        ]
    )
    by_name = {row.name: row for row in rows}
    assert by_name["stairs"].instances == 1
    assert by_name["stairs"].status != STATUS_GOOD
