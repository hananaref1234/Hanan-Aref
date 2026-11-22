"""Review state machine, including directional-label protection."""

from __future__ import annotations

import pytest

from src.dataset.importers.provenance import ProvenanceRecord
from src.dataset.importers.review import (
    ReviewError,
    apply_keyboard_key,
    apply_review_action,
    bulk_approve_pending_semantic,
    class_for_promotion,
    review_mode_for_record,
)


def _record(**kwargs) -> ProvenanceRecord:
    base = dict(
        sample_id="s1",
        source_dataset="accessibility_barriers",
        source_image="a.jpg",
        source_split="train",
        original_class="stairs",
        mapped_class="",
        mapping_type="MANUAL_DIRECTION",
        review_status="NEEDS_DIRECTION_REVIEW",
        sequence_id="seq1",
        license="CC-BY-4.0",
        proposed_class="",
        original_annotation="xyxy=1,1,10,10;wh=32x32",
        staging_image="a.jpg",
        staging_label="a.txt",
    )
    base.update(kwargs)
    return ProvenanceRecord(**base)


def test_directional_keys_assign_classes() -> None:
    record = _record()
    up = apply_keyboard_key(record, "1")
    assert up.review_status == "STAIRS_UP"
    assert up.mapped_class == "stairs"
    assert up.direction_status == "STAIRS_UP"
    assert up.direction_class == "stairs_up"
    assert class_for_promotion(up) == "stairs"

    down = apply_keyboard_key(record, "2")
    assert down.mapped_class == "stairs"
    assert down.direction_class == "stairs_down"
    curb_up = apply_keyboard_key(record, "3")
    assert curb_up.mapped_class == "curb"
    assert curb_up.direction_class == "curb_up"


def test_ambiguous_and_reject_are_not_promotable() -> None:
    record = _record()
    rejected = apply_keyboard_key(record, "r")
    assert class_for_promotion(rejected) is None
    ambiguous = apply_keyboard_key(record, "x")
    assert class_for_promotion(ambiguous) is None
    assert ambiguous.mapped_class == ""


def test_cannot_approve_without_a_class() -> None:
    record = _record()
    with pytest.raises(ReviewError, match="no mapped or proposed class"):
        apply_review_action(record, "APPROVED")


def test_direct_pending_can_be_approved() -> None:
    record = _record(
        original_class="pole",
        mapped_class="pole",
        proposed_class="pole",
        mapping_type="DIRECT",
        review_status="PENDING",
        source_dataset="sideguide",
    )
    approved = apply_keyboard_key(record, "a")
    assert approved.review_status == "APPROVED"
    assert class_for_promotion(approved) == "pole"


def test_cannot_assign_direction_to_direct_pole() -> None:
    record = _record(
        original_class="pole",
        mapped_class="pole",
        mapping_type="DIRECT",
        review_status="PENDING",
    )
    with pytest.raises(ReviewError, match="Cannot assign STAIRS_UP"):
        apply_review_action(record, "STAIRS_UP")


def test_r_rejects_without_guessing_direction() -> None:
    record = _record()
    rejected = apply_keyboard_key(record, "r")
    assert rejected.review_status == "REJECTED"
    assert rejected.mapped_class == ""
    assert class_for_promotion(rejected) is None


def test_unknown_key_rejected() -> None:
    with pytest.raises(ReviewError, match="Unknown key"):
        apply_keyboard_key(_record(), "z")


def test_semantic_reject_does_not_overwrite_direction_metadata() -> None:
    record = _record(
        original_class="pole",
        mapped_class="pole",
        proposed_class="pole",
        mapping_type="DIRECT",
        review_status="PENDING",
        direction_status="STAIRS_UP",
        direction_class="stairs_up",
    )
    rejected = apply_keyboard_key(record, "r", mode="semantic")
    assert rejected.review_status == "REJECTED"
    assert rejected.direction_status == "STAIRS_UP"
    assert rejected.direction_class == "stairs_up"


def test_semantic_mode_allows_approve_and_reject_only() -> None:
    record = _record(
        original_class="bollard",
        mapped_class="bollard",
        proposed_class="bollard",
        mapping_type="DIRECT",
        review_status="PENDING",
    )
    approved = apply_keyboard_key(record, "a", mode="semantic")
    assert approved.review_status == "APPROVED"
    with pytest.raises(ReviewError, match="not valid for semantic review"):
        apply_keyboard_key(record, "1", mode="semantic")


def test_review_mode_detects_direction_and_semantic_rows() -> None:
    assert review_mode_for_record(_record()) == "direction"
    semantic = _record(
        original_class="tree_trunk",
        mapped_class="tree_trunk",
        proposed_class="tree_trunk",
        mapping_type="DIRECT",
        review_status="PENDING",
    )
    assert review_mode_for_record(semantic) == "semantic"


def test_manual_approve_sets_review_method() -> None:
    record = _record(
        original_class="bollard",
        mapped_class="bollard",
        proposed_class="bollard",
        mapping_type="DIRECT",
        review_status="PENDING",
    )
    approved = apply_keyboard_key(record, "a", mode="semantic")
    assert approved.review_method == "MANUAL_REVIEW"


def test_bulk_approve_preserves_human_decisions() -> None:
    pending = _record(
        sample_id="bulk-bollard",
        original_class="bollard",
        mapped_class="bollard",
        proposed_class="bollard",
        mapping_type="DIRECT",
        review_status="PENDING",
    )
    manual = _record(
        sample_id="manual-bollard",
        original_class="bollard",
        mapped_class="bollard",
        proposed_class="bollard",
        mapping_type="DIRECT",
        review_status="APPROVED",
        review_method="MANUAL_REVIEW",
    )
    rejected = _record(
        sample_id="rejected-barrier",
        original_class="barricade",
        mapped_class="barrier",
        proposed_class="barrier",
        mapping_type="DIRECT",
        review_status="REJECTED",
    )
    pole_pending = _record(
        sample_id="pending-pole",
        original_class="pole",
        mapped_class="pole",
        proposed_class="pole",
        mapping_type="DIRECT",
        review_status="PENDING",
    )
    updated, counts = bulk_approve_pending_semantic(
        [pending, manual, rejected, pole_pending],
        kinds=("bollard", "tree_trunk", "barrier"),
    )
    by_id = {record.sample_id: record for record in updated}
    assert counts == {"bollard": 1, "tree_trunk": 0, "barrier": 0}
    assert by_id["bulk-bollard"].review_method == "SOURCE_MAPPING_BULK_APPROVAL"
    assert by_id["manual-bollard"].review_method == "MANUAL_REVIEW"
    assert by_id["rejected-barrier"].review_status == "REJECTED"
    assert by_id["pending-pole"].review_status == "PENDING"
