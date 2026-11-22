"""Preserve taxonomy v0.1 human direction decisions across a v0.2 re-import."""

from __future__ import annotations

from dataclasses import replace

from src.dataset.importers.provenance import ProvenanceRecord
from src.dataset.importers.review import DIRECTION_ACTIONS, _join_notes
from src.dataset.taxonomy import DIRECTIONAL_CLASSES_V0_1, detector_class_from_direction

DIRECTION_STATUSES = frozenset(DIRECTION_ACTIONS)


def preserve_v01_direction_metadata(
    new_records: list[ProvenanceRecord],
    old_records: list[ProvenanceRecord],
) -> list[ProvenanceRecord]:
    """Copy STAIRS_*/CURB_* / REJECTED decisions onto rebuilt v0.2 rows.

    Detector `mapped_class` stays semantic (`stairs` / `curb`) except REJECTED
    rows, which are kept out of promotion.
    """
    by_id = {record.sample_id: record for record in old_records if record.sample_id}
    by_key = {
        _match_key(record): record
        for record in old_records
        if record.source_image and record.original_annotation
    }
    updated: list[ProvenanceRecord] = []
    for record in new_records:
        old = by_id.get(record.sample_id) or by_key.get(_match_key(record))
        if old is None:
            updated.append(record)
            continue
        updated.append(_merge_one(record, old))
    return updated


def _match_key(record: ProvenanceRecord) -> tuple[str, str]:
    return (record.source_image, record.original_annotation)


def _merge_one(new: ProvenanceRecord, old: ProvenanceRecord) -> ProvenanceRecord:
    old_status = (old.review_status or "").upper()
    old_direction = old.direction_class
    if old_direction not in DIRECTIONAL_CLASSES_V0_1:
        if old.mapped_class in DIRECTIONAL_CLASSES_V0_1:
            old_direction = old.mapped_class
        else:
            old_direction = DIRECTION_ACTIONS.get(old_status, "")
    if old_status in DIRECTION_STATUSES:
        semantic = detector_class_from_direction(old_direction) or new.mapped_class
        return replace(
            new,
            review_status=old_status,
            mapped_class=semantic,
            proposed_class=semantic,
            direction_status=old_status,
            direction_class=old_direction,
            notes=_join_notes(
                new.notes,
                f"Preserved v0.1 direction {old_status} ({old_direction}).",
            ),
        )
    if old_status == "REJECTED":
        return replace(
            new,
            review_status="REJECTED",
            mapped_class="",
            direction_status="REJECTED",
            direction_class=old_direction,
            notes=_join_notes(new.notes, "Preserved v0.1 REJECTED decision."),
        )
    if old_status == "AMBIGUOUS":
        return replace(
            new,
            review_status="AMBIGUOUS",
            mapped_class="",
            direction_status="AMBIGUOUS",
            notes=_join_notes(new.notes, "Preserved v0.1 AMBIGUOUS decision."),
        )
    return new
