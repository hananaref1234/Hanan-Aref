"""Local review state machine for staging samples.

This is not a web app. Keyboard bindings for the OpenCV reviewer live in
scripts/review_staging.py and docs/STAGING_REVIEW.md.

Taxonomy v0.2: direction keys write research metadata. Detector promotion uses
semantic `stairs` / `curb` only. Direction review is optional, not required
for object-detector training.
"""

from __future__ import annotations

from dataclasses import replace

from src.dataset.importers.provenance import ProvenanceRecord
from src.dataset.taxonomy import (
    CUSTOM_CLASSES,
    DIRECTION_TO_DETECTOR_CLASS,
    detector_class_from_direction,
)

REVIEW_METHOD_MANUAL = "MANUAL_REVIEW"
REVIEW_METHOD_BULK = "SOURCE_MAPPING_BULK_APPROVAL"

BULK_APPROVE_SEMANTIC_KINDS = ("bollard", "tree_trunk", "barrier")

REVIEW_STATUSES = (
    "PENDING",
    "APPROVED",
    "REJECTED",
    "NEEDS_DIRECTION_REVIEW",
    "STAIRS_UP",
    "STAIRS_DOWN",
    "CURB_UP",
    "CURB_DOWN",
    "AMBIGUOUS",
    "UNMAPPED_SOURCE_CLASS",
)

DIRECTION_ACTIONS = {
    "STAIRS_UP": "stairs_up",
    "STAIRS_DOWN": "stairs_down",
    "CURB_UP": "curb_up",
    "CURB_DOWN": "curb_down",
}

PROMOTABLE_STATUSES = frozenset(
    {
        "APPROVED",
        "STAIRS_UP",
        "STAIRS_DOWN",
        "CURB_UP",
        "CURB_DOWN",
    }
)

KEYBOARD_ACTIONS = {
    "a": "APPROVED",
    "r": "REJECTED",
    "1": "STAIRS_UP",
    "2": "STAIRS_DOWN",
    "3": "CURB_UP",
    "4": "CURB_DOWN",
    "x": "AMBIGUOUS",
}


class ReviewError(ValueError):
    """Invalid review transition."""


def review_mode_for_record(record: ProvenanceRecord) -> str:
    """Return `direction` for stairs/curb metadata rows, else `semantic`."""
    original = record.original_class.lower().strip()
    proposed = (record.proposed_class or record.mapped_class).lower().strip()
    if original in {"stairs", "stair", "step", "steps", "curb", "kerb"}:
        return "direction"
    if proposed in {"stairs", "curb"} and record.mapping_type in {
        "MANUAL_DIRECTION",
        "DIRECT",
        "SEMANTIC_REVIEW",
        "CUSTOM",
    }:
        return "direction"
    return "semantic"


def apply_review_action(record: ProvenanceRecord, action: str) -> ProvenanceRecord:
    """Apply a reviewer decision. Directional labels are never inferred."""
    action = action.strip().upper()
    if action not in REVIEW_STATUSES:
        raise ReviewError(
            f"Unknown review action {action!r}. "
            f"Allowed: {', '.join(REVIEW_STATUSES)}."
        )
    if action == "PENDING":
        return replace(record, review_status="PENDING")
    if action == "REJECTED":
        return replace(
            record,
            review_status="REJECTED",
            mapped_class="",
        )
    if action == "AMBIGUOUS":
        return replace(
            record,
            review_status="AMBIGUOUS",
            mapped_class="",
            notes=_join_notes(record.notes, "Reviewer marked AMBIGUOUS."),
        )
    if action in DIRECTION_ACTIONS:
        semantic_ok = (
            record.mapped_class in {"stairs", "curb"}
            or record.proposed_class in {"stairs", "curb"}
            or record.original_class.lower() in {"stair", "stairs", "step", "steps", "curb", "kerb"}
        )
        if record.mapping_type not in {
            "MANUAL_DIRECTION",
            "CUSTOM",
            "SEMANTIC_REVIEW",
            "DIRECT",
        }:
            raise ReviewError(
                f"Cannot assign {action} to a {record.mapping_type} sample."
            )
        if record.mapping_type == "DIRECT" and not semantic_ok:
            raise ReviewError(
                f"Cannot assign {action} to a {record.mapping_type} sample."
            )
        direction_class = DIRECTION_ACTIONS[action]
        semantic = detector_class_from_direction(direction_class) or ""
        return replace(
            record,
            review_status=action,
            mapped_class=semantic,
            proposed_class=semantic or record.proposed_class,
            direction_status=action,
            direction_class=direction_class,
            notes=_join_notes(
                record.notes,
                f"Human assigned {direction_class} as direction metadata; "
                f"detector class is {semantic}.",
            ),
        )
    if action == "APPROVED":
        mapped = record.mapped_class or record.proposed_class
        if mapped in DIRECTION_TO_DETECTOR_CLASS:
            mapped = DIRECTION_TO_DETECTOR_CLASS[mapped]
        if not mapped:
            raise ReviewError(
                "Cannot APPROVE a sample with no mapped or proposed class. "
                "Assign a detector class or reject it."
            )
        if mapped not in CUSTOM_CLASSES and record.taxonomy != "combined":
            raise ReviewError(
                f"Cannot APPROVE {mapped!r} into the custom taxonomy."
            )
        return replace(
            record,
            review_status="APPROVED",
            mapped_class=mapped,
            review_method=REVIEW_METHOD_MANUAL,
        )
    if action == "NEEDS_DIRECTION_REVIEW":
        return replace(
            record,
            review_status="NEEDS_DIRECTION_REVIEW",
            mapped_class="",
        )
    if action == "UNMAPPED_SOURCE_CLASS":
        return replace(
            record,
            review_status="UNMAPPED_SOURCE_CLASS",
            mapped_class="",
            proposed_class="",
        )
    raise ReviewError(f"Unhandled review action {action!r}.")


def apply_keyboard_key(
    record: ProvenanceRecord,
    key: str,
    *,
    mode: str | None = None,
) -> ProvenanceRecord:
    """Map a reviewer keypress to a review action."""
    action = KEYBOARD_ACTIONS.get(key.lower())
    if action is None:
        raise ReviewError(
            f"Unknown key {key!r}. Valid keys: a r 1 2 3 4 x "
            f"(see docs/STAGING_REVIEW.md)."
        )
    resolved_mode = mode or review_mode_for_record(record)
    if resolved_mode == "semantic" and action in {"STAIRS_UP", "STAIRS_DOWN", "CURB_UP", "CURB_DOWN", "AMBIGUOUS"}:
        raise ReviewError(
            f"Key {key!r} is not valid for semantic review. "
            "Use: a approve, r reject, s/n skip, p previous, q quit, h help."
        )
    return apply_review_action(record, action)


def class_for_promotion(record: ProvenanceRecord) -> str | None:
    """Return the v0.2 detector class if this record may enter the training set."""
    if record.review_status in {"REJECTED", "AMBIGUOUS", "UNMAPPED_SOURCE_CLASS", "PENDING"}:
        return None
    if record.review_status == "NEEDS_DIRECTION_REVIEW":
        return None
    if record.review_status in DIRECTION_ACTIONS:
        return detector_class_from_direction(DIRECTION_ACTIONS[record.review_status])
    mapped = (record.mapped_class or "").strip()
    if mapped in DIRECTION_TO_DETECTOR_CLASS:
        mapped = DIRECTION_TO_DETECTOR_CLASS[mapped]
    if mapped in CUSTOM_CLASSES and record.review_status in PROMOTABLE_STATUSES:
        return mapped
    return None


def record_matches_semantic_kind(record: ProvenanceRecord, kind: str) -> bool:
    """Return True when a staging row belongs to a semantic review kind."""
    original = record.original_class.lower()
    proposed = (record.proposed_class or record.mapped_class).lower()
    if kind == "pole":
        return original == "pole" or proposed == "pole"
    if kind == "bollard":
        return original == "bollard" or proposed == "bollard"
    if kind == "tree_trunk":
        return original == "tree_trunk" or proposed == "tree_trunk"
    if kind == "barrier":
        return original == "barricade" or proposed == "barrier"
    raise ReviewError(f"Unknown semantic kind {kind!r}.")


def bulk_approve_pending_semantic(
    records: list[ProvenanceRecord],
    kinds: tuple[str, ...] = BULK_APPROVE_SEMANTIC_KINDS,
) -> tuple[list[ProvenanceRecord], dict[str, int]]:
    """Approve remaining PENDING rows for selected semantic kinds.

    Existing APPROVED and REJECTED rows are preserved. Each bulk-approved row
    records ``review_method=SOURCE_MAPPING_BULK_APPROVAL``.
    """
    approved_by_kind: dict[str, int] = {kind: 0 for kind in kinds}
    updated: list[ProvenanceRecord] = []
    for record in records:
        if record.review_status != "PENDING":
            updated.append(record)
            continue
        matched_kind = next(
            (kind for kind in kinds if record_matches_semantic_kind(record, kind)),
            None,
        )
        if matched_kind is None:
            updated.append(record)
            continue
        approved = apply_review_action(record, "APPROVED")
        approved = replace(
            approved,
            review_method=REVIEW_METHOD_BULK,
            notes=_join_notes(
                approved.notes,
                "Bulk-approved pending semantic mapping after manual spot-check.",
            ),
        )
        updated.append(approved)
        approved_by_kind[matched_kind] += 1
    return updated, approved_by_kind


def semantic_review_stats(
    records: list[ProvenanceRecord],
    kind: str,
) -> dict[str, int]:
    """Count review_status rows for one semantic kind."""
    matched = [record for record in records if record_matches_semantic_kind(record, kind)]
    stats = {
        "reviewed": 0,
        "approved": 0,
        "rejected": 0,
        "pending": 0,
    }
    for record in matched:
        status = record.review_status
        if status == "PENDING":
            stats["pending"] += 1
        elif status == "REJECTED":
            stats["reviewed"] += 1
            stats["rejected"] += 1
        elif status in PROMOTABLE_STATUSES or status == "APPROVED":
            stats["reviewed"] += 1
            stats["approved"] += 1
    return stats


def _join_notes(existing: str, extra: str) -> str:
    existing = existing.strip()
    if not existing:
        return extra
    if extra in existing:
        return existing
    return f"{existing} {extra}"
