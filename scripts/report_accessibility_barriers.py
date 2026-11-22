"""Summarize Accessibility Barriers raw + staging counts. Does not train."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers.duplicates import find_exact_image_duplicates
from src.dataset.importers.provenance import default_metadata_path, read_provenance
from src.dataset.importers.source_inventory import inventory_accessibility_barriers
from src.dataset.manifest import get_source
from src.utils.paths import project_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    args = parser.parse_args()
    source = get_source("accessibility_barriers")
    inventory = inventory_accessibility_barriers(source.raw_path)
    records = read_provenance(default_metadata_path(project_root() / "data" / "staging"))
    ab = [row for row in records if row.source_dataset == "accessibility_barriers"]
    status = Counter(row.review_status for row in ab)
    stairs = [
        row
        for row in ab
        if row.original_class.lower() in {"stairs", "stair"}
        and row.review_status == "NEEDS_DIRECTION_REVIEW"
    ]
    curbs = [
        row
        for row in ab
        if row.review_status == "NEEDS_DIRECTION_REVIEW"
        and (
            row.original_class.lower() in {"curb", "kerb"}
            or "curb-type" in row.notes.lower()
        )
    ]
    unmapped = [row for row in ab if row.review_status == "UNMAPPED_SOURCE_CLASS"]
    unmapped_classes = Counter()
    for row in unmapped:
        key = row.original_class
        if row.original_class.lower() == "step":
            if "type=doorstep" in row.notes.lower() or "doorstep" in row.notes.lower():
                key = "step/doorstep"
            elif "type=other" in row.notes.lower():
                key = "step/other"
            elif "generic step" in row.notes.lower():
                key = "step/generic"
            else:
                key = "step"
        unmapped_classes[key] += 1
    duplicates = find_exact_image_duplicates(
        [row for row in ab if row.review_status == "NEEDS_DIRECTION_REVIEW"]
    )
    payload = {
        "raw_images_on_disk": inventory.images_on_disk,
        "xml_images": inventory.images_in_xml,
        "xml_missing_images": inventory.missing_images,
        "xml_annotations": inventory.annotations,
        "xml_labels": dict(inventory.labels),
        "xml_step_types": dict(inventory.step_types),
        "staged_records": len(ab),
        "staircase_candidates": len(stairs),
        "curb_candidates": len(curbs),
        "unmapped": dict(unmapped_classes),
        "review_status": dict(status),
        "approved_stairs_up": status.get("STAIRS_UP", 0),
        "approved_stairs_down": status.get("STAIRS_DOWN", 0),
        "approved_curb_up": status.get("CURB_UP", 0),
        "approved_curb_down": status.get("CURB_DOWN", 0),
        "rejected": status.get("REJECTED", 0),
        "ambiguous": status.get("AMBIGUOUS", 0),
        "pending_direction_review": status.get("NEEDS_DIRECTION_REVIEW", 0),
        "exact_duplicate_image_groups": len(duplicates),
        "exact_duplicate_images": sum(len(group.source_images) for group in duplicates),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print("Accessibility Barriers import summary")
    print(f"  XML images: {payload['xml_images']}")
    print(f"  Images on disk: {payload['raw_images_on_disk']}")
    print(f"  XML annotations: {payload['xml_annotations']}")
    print(f"  Labels: {payload['xml_labels']}")
    print(f"  Step types: {payload['xml_step_types']}")
    print(f"  Staged records: {payload['staged_records']}")
    print(f"  Staircase candidates: {payload['staircase_candidates']}")
    print(f"  Curb candidates: {payload['curb_candidates']}")
    print(f"  Unmapped: {payload['unmapped']}")
    print(f"  Review status: {payload['review_status']}")
    print(
        "  Approved: "
        f"stairs_up={payload['approved_stairs_up']} "
        f"stairs_down={payload['approved_stairs_down']} "
        f"curb_up={payload['approved_curb_up']} "
        f"curb_down={payload['approved_curb_down']}"
    )
    print(
        f"  Rejected={payload['rejected']} ambiguous={payload['ambiguous']} "
        f"pending={payload['pending_direction_review']}"
    )
    print(
        f"  Exact duplicate image groups: {payload['exact_duplicate_image_groups']} "
        f"({payload['exact_duplicate_images']} images)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
