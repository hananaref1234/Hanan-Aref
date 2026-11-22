"""Review queue and reporting helpers."""

from __future__ import annotations

from src.dataset.importers.provenance import ProvenanceRecord
from scripts.review_staging import annotate_record, build_queue, review_stats


def _record(**kwargs) -> ProvenanceRecord:
    base = dict(
        sample_id="s1",
        source_dataset="roboflow_sidewalk_barriers",
        source_image="train/images/a.jpg",
        source_split="train",
        original_class="pole",
        mapped_class="pole",
        mapping_type="DIRECT",
        review_status="PENDING",
        sequence_id="seq1",
        license="CC-BY-4.0",
        proposed_class="pole",
        original_annotation="xyxy=1,1,10,10;wh=32x32",
        staging_image="missing.jpg",
        staging_label="a.txt",
        notes="Direct mapping candidate; sample remains PENDING until review.",
    )
    base.update(kwargs)
    return ProvenanceRecord(**base)


def test_already_decided_rows_omitted_from_pending_queue() -> None:
    records = [
        _record(sample_id="pending", review_status="PENDING"),
        _record(sample_id="approved", review_status="APPROVED"),
        _record(sample_id="rejected", review_status="REJECTED"),
    ]
    queue = build_queue(records, statuses={"PENDING"}, kind="pole", limit=None)
    assert [row.sample_id for row in queue] == ["pending"]


def test_kind_filters_semantic_barrier_queue() -> None:
    records = [
        _record(sample_id="bar", original_class="barricade", proposed_class="barrier", mapped_class=""),
        _record(sample_id="pole", original_class="pole", proposed_class="pole"),
    ]
    queue = build_queue(records, statuses={"PENDING"}, kind="barrier", limit=None)
    assert [row.sample_id for row in queue] == ["bar"]


def test_review_stats_reports_approval_counts() -> None:
    records = [
        _record(sample_id="a1", original_class="pole", review_status="APPROVED"),
        _record(sample_id="a2", original_class="pole", review_status="APPROVED"),
        _record(sample_id="r1", original_class="pole", review_status="REJECTED"),
        _record(sample_id="p1", original_class="pole", review_status="PENDING"),
    ]
    stats = review_stats(records, kind="pole")
    assert stats[0]["source_class"] == "pole"
    assert stats[0]["approved"] == 2
    assert stats[0]["rejected"] == 1
    assert stats[0]["pending"] == 1


def test_annotate_record_shows_semantic_overlay() -> None:
    frame = annotate_record(_record(), index=41, total=2000)
    assert frame is not None


def test_stairs_direction_kind_still_works() -> None:
    record = _record(
        original_class="stairs",
        proposed_class="stairs",
        mapped_class="",
        mapping_type="MANUAL_DIRECTION",
        review_status="NEEDS_DIRECTION_REVIEW",
        notes="stairs sample",
    )
    queue = build_queue([record], statuses={"NEEDS_DIRECTION_REVIEW"}, kind="stairs", limit=None)
    assert [row.sample_id for row in queue] == ["s1"]
