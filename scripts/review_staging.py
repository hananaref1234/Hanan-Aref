"""Lightweight OpenCV reviewer for staging samples.

Optional research tool. Direction review is NOT required for taxonomy v0.2
object-detector training.

Direction review keys (`stairs`, `curb`):
  a  APPROVED (semantic detector class only)
  1  STAIRS_UP direction metadata (detector class stays stairs)
  2  STAIRS_DOWN direction metadata (detector class stays stairs)
  3  CURB_UP direction metadata (detector class stays curb)
  4  CURB_DOWN direction metadata (detector class stays curb)
  r  reject
  x  ambiguous

Semantic review keys (`pole`, `bollard`, `tree_trunk`, `barrier`):
  a  APPROVED
  r  REJECTED

Shared navigation keys:
  s / n  skip / next
  p  previous
  q  quit and save
  h  help

Reviewed rows are written to staging_metadata.csv immediately, so you can
stop and continue later. Already-decided samples are omitted from the queue.

Does not invent directional labels. See docs/STAGING_REVIEW.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers.boxes import PixelBox
from src.dataset.importers.provenance import (
    ProvenanceRecord,
    default_metadata_path,
    read_provenance,
    write_provenance,
)
from src.dataset.importers.review import ReviewError, apply_keyboard_key, review_mode_for_record
from src.utils.paths import project_root

KIND_CHOICES = ("all", "stairs", "curb", "pole", "bollard", "tree_trunk", "barrier")


def _in_kind(record: ProvenanceRecord, kind: str) -> bool:
    if kind == "all":
        return True
    original = record.original_class.lower()
    proposed = (record.proposed_class or record.mapped_class).lower()
    notes = record.notes.lower()
    if kind == "stairs":
        return original in {"stairs", "stair"}
    if kind == "curb":
        return original in {"curb", "kerb"} or "curb-type" in notes or "curb_up" in notes
    if kind == "pole":
        return original == "pole" or proposed == "pole"
    if kind == "bollard":
        return original == "bollard" or proposed == "bollard"
    if kind == "tree_trunk":
        return original == "tree_trunk" or proposed == "tree_trunk"
    if kind == "barrier":
        return original == "barricade" or proposed == "barrier"
    return False


def build_queue(
    records: list[ProvenanceRecord],
    *,
    statuses: set[str],
    kind: str,
    limit: int | None,
) -> list[ProvenanceRecord]:
    queue = [
        record
        for record in records
        if record.review_status.upper() in statuses and _in_kind(record, kind)
    ]
    if limit is not None:
        queue = queue[: max(0, limit)]
    return queue


def review_stats(records: list[ProvenanceRecord], *, kind: str) -> list[dict[str, object]]:
    groups: dict[str, dict[str, int]] = {}
    for record in records:
        if not _in_kind(record, kind):
            continue
        label = record.original_class
        bucket = groups.setdefault(label, {"APPROVED": 0, "REJECTED": 0, "PENDING": 0, "OTHER": 0})
        status = record.review_status.upper()
        if status == "APPROVED":
            bucket["APPROVED"] += 1
        elif status == "REJECTED":
            bucket["REJECTED"] += 1
        elif status == "PENDING":
            bucket["PENDING"] += 1
        else:
            bucket["OTHER"] += 1
    rows: list[dict[str, object]] = []
    for label in sorted(groups):
        bucket = groups[label]
        decided = bucket["APPROVED"] + bucket["REJECTED"]
        denom = decided + bucket["PENDING"]
        approval_pct = (bucket["APPROVED"] / decided * 100.0) if decided else 0.0
        rows.append(
            {
                "source_class": label,
                "approved": bucket["APPROVED"],
                "rejected": bucket["REJECTED"],
                "pending": bucket["PENDING"],
                "other": bucket["OTHER"],
                "approval_pct": approval_pct,
                "decision_coverage_pct": (decided / denom * 100.0) if denom else 0.0,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        default=str(project_root() / "data" / "staging"),
    )
    parser.add_argument(
        "--filter-status",
        default="PENDING,NEEDS_DIRECTION_REVIEW",
        help="Comma-separated review_status values to show.",
    )
    parser.add_argument(
        "--kind",
        choices=KIND_CHOICES,
        default="all",
        help=(
            "Restrict the queue to one review kind: direction (`stairs`, `curb`) "
            "or semantic (`pole`, `bollard`, `tree_trunk`, `barrier`)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Review at most N queued samples this sitting.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the queue and exit (no GUI).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print approved/rejected/pending counts and approval percentage for the selected kind.",
    )
    args = parser.parse_args()

    metadata_path = default_metadata_path(Path(args.staging_root))
    records = read_provenance(metadata_path)
    wanted = {item.strip().upper() for item in args.filter_status.split(",") if item.strip()}
    full_queue = build_queue(records, statuses=wanted, kind=args.kind, limit=None)
    queue = full_queue[: args.limit] if args.limit is not None else full_queue
    print(
        f"{len(full_queue)} samples pending review "
        f"(showing {len(queue)}; kind={args.kind}, status={','.join(sorted(wanted))}) "
        f"({metadata_path})"
    )
    if args.stats:
        for row in review_stats(records, kind=args.kind):
            print(
                f"{row['source_class']}\tapproved={row['approved']}\t"
                f"rejected={row['rejected']}\tpending={row['pending']}\t"
                f"approval={row['approval_pct']:.1f}%"
            )
        return 0
    if args.list or not queue:
        for record in queue[:50]:
            print(
                f"{record.sample_id}\t{record.source_dataset}\t"
                f"{record.original_class}\t{record.proposed_class or '-'}\t"
                f"{record.review_status}"
            )
        if not queue:
            return 0
        if args.list:
            return 0

    import cv2

    index = 0
    while 0 <= index < len(queue):
        record = queue[index]
        if record.review_status.upper() not in wanted:
            index += 1
            continue
        frame = annotate_record(record, index=index, total=len(queue))
        cv2.imshow("Li-Stick staging review", frame)
        key = cv2.waitKey(0) & 0xFF
        char = chr(key) if 32 <= key < 127 else ""
        if char in {"q"}:
            break
        if char in {"n", "s"}:
            index += 1
            continue
        if char in {"p"}:
            index = max(0, index - 1)
            continue
        if char in {"h"}:
            print(__doc__)
            continue
        mode = review_mode_for_record(record)
        try:
            updated = apply_keyboard_key(record, char, mode=mode)
        except ReviewError as exc:
            print(exc)
            continue
        _replace(records, updated)
        queue[index] = updated
        write_provenance(metadata_path, records)
        print(f"{updated.sample_id} -> {updated.review_status} {updated.mapped_class}")
        index += 1
    cv2.destroyAllWindows()
    write_provenance(metadata_path, records)
    return 0


def _replace(records, updated) -> None:
    for i, record in enumerate(records):
        if record.sample_id == updated.sample_id:
            records[i] = updated
            return


def annotate_record(record: ProvenanceRecord, *, index: int = 0, total: int = 0):
    """Draw source box and review instructions. Used by tests without a window."""
    import cv2
    import numpy as np

    image = cv2.imread(record.staging_image)
    if image is None:
        image = np.zeros((360, 640, 3), dtype=np.uint8)
    progress = f"sample {index + 1} / {total}" if total else ""
    mode = review_mode_for_record(record)
    controls = _controls_line(mode)
    overlay = [
        f"dataset: {record.source_dataset}   {progress}",
        f"source: {record.original_class}",
        f"proposed: {record.proposed_class or record.mapped_class or '-'}",
        f"status: {record.review_status}   mapping: {record.mapping_type}   mode: {mode}",
        f"split: {record.source_split or '-'}   seq: {record.sequence_id or '-'}",
        record.notes[:110],
        controls,
    ]
    y = 28
    for line in overlay:
        cv2.putText(
            image,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    box = _box_from_record(record)
    if box is not None:
        pt1 = (int(box.x1), int(box.y1))
        pt2 = (int(box.x2), int(box.y2))
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
        cv2.putText(
            image,
            record.original_class,
            (pt1[0], max(20, pt1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def _controls_line(mode: str) -> str:
    if mode == "semantic":
        return "a approve  r reject  s/n skip  p prev  q quit  h help"
    return "a approve  1 stairs_up  2 stairs_down  3 curb_up  4 curb_down  r reject  x ambiguous  s/n skip"


def _box_from_record(record):
    text = record.original_annotation
    if "xyxy=" not in text:
        return None
    xyxy_part = text.split("xyxy=", 1)[1].split(";", 1)[0]
    x1, y1, x2, y2 = (float(part) for part in xyxy_part.split(","))
    return PixelBox(x1, y1, x2, y2)


if __name__ == "__main__":
    raise SystemExit(main())
