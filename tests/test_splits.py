"""Seeded image-level splits and v0.1 direction preservation."""

from __future__ import annotations

from src.dataset.importers.provenance import ProvenanceRecord
from src.dataset.importers.review import class_for_promotion
from src.dataset.importers.splits import assign_splits_by_source_image, split_counts
from src.dataset.importers.v01_preserve import preserve_v01_direction_metadata


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
        "original_annotation": "xyxy=1,1,10,10;wh=32x32",
        "taxonomy_version": "v0.2",
    }
    data.update(overrides)
    return ProvenanceRecord(**data)


def test_same_source_image_stays_in_one_split() -> None:
    records = [
        _record(sample_id="a0", source_image="images/a.jpg"),
        _record(sample_id="a1", source_image="images/a.jpg"),
        _record(sample_id="b0", source_image="images/b.jpg"),
        _record(sample_id="c0", source_image="images/c.jpg"),
        _record(sample_id="d0", source_image="images/d.jpg"),
        _record(sample_id="e0", source_image="images/e.jpg"),
    ]
    assigned = assign_splits_by_source_image(records, seed=42)
    by_image: dict[str, set[str]] = {}
    for record in assigned:
        by_image.setdefault(record.source_image, set()).add(record.detector_split)
    assert all(len(splits) == 1 for splits in by_image.values())
    counts = split_counts(assigned)
    assert counts["train"]["images"] + counts["val"]["images"] + counts["test"]["images"] == 5
    again = assign_splits_by_source_image(records, seed=42)
    assert [row.detector_split for row in again] == [row.detector_split for row in assigned]


def test_existing_source_split_is_kept() -> None:
    records = [
        _record(sample_id="t", source_image="a.jpg", source_split="train"),
        _record(sample_id="v", source_image="b.jpg", source_split="val"),
    ]
    assigned = assign_splits_by_source_image(records, seed=1)
    assert assigned[0].detector_split == "train"
    assert assigned[1].detector_split == "val"


def test_v01_direction_is_preserved_as_metadata() -> None:
    old = [
        _record(
            sample_id="accessibility_barriers_unsplit_a_0000",
            source_image="images/a.jpg",
            mapped_class="stairs_up",
            mapping_type="MANUAL_DIRECTION",
            review_status="STAIRS_UP",
        )
    ]
    new = [
        _record(
            sample_id="accessibility_barriers_unsplit_a_0000",
            source_image="images/a.jpg",
            mapped_class="stairs",
            review_status="APPROVED",
        )
    ]
    merged = preserve_v01_direction_metadata(new, old)
    assert merged[0].direction_status == "STAIRS_UP"
    assert merged[0].direction_class == "stairs_up"
    assert merged[0].mapped_class == "stairs"
    assert class_for_promotion(merged[0]) == "stairs"


def test_v01_rejected_is_not_promotable() -> None:
    old = [
        _record(
            sample_id="s1",
            review_status="REJECTED",
            mapped_class="",
        )
    ]
    new = [_record(sample_id="s1", review_status="APPROVED", mapped_class="stairs")]
    merged = preserve_v01_direction_metadata(new, old)
    assert merged[0].review_status == "REJECTED"
    assert class_for_promotion(merged[0]) is None
