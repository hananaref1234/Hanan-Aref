"""Bulk-approve pending semantic staging rows after manual spot-checks.

Does not promote, train, or overwrite prior human APPROVED/REJECTED rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers.provenance import (
    default_metadata_path,
    read_provenance,
    write_provenance,
)
from src.dataset.importers.review import (
    BULK_APPROVE_SEMANTIC_KINDS,
    bulk_approve_pending_semantic,
    semantic_review_stats,
)
from src.utils.paths import project_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        default=str(project_root() / "data" / "staging"),
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=list(BULK_APPROVE_SEMANTIC_KINDS),
        choices=["bollard", "tree_trunk", "barrier", "pole"],
        help="Semantic kinds to bulk-approve (default: bollard tree_trunk barrier).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = default_metadata_path(Path(args.staging_root))
    records = read_provenance(metadata)
    updated, approved_by_kind = bulk_approve_pending_semantic(records, tuple(args.kinds))

    print("Bulk approval summary:")
    for kind in args.kinds:
        count = approved_by_kind.get(kind, 0)
        print(f"  {kind}: {count} pending -> APPROVED (SOURCE_MAPPING_BULK_APPROVAL)")
    print(f"  total bulk-approved: {sum(approved_by_kind.values())}")

    if "pole" not in args.kinds:
        pole = semantic_review_stats(updated, "pole")
        reviewed = pole["reviewed"]
        approval_pct = (
            f"{100.0 * pole['approved'] / reviewed:.1f}%" if reviewed else "n/a"
        )
        print(
            "Pole unchanged: "
            f"reviewed={reviewed} approved={pole['approved']} "
            f"rejected={pole['rejected']} pending={pole['pending']} "
            f"approval={approval_pct}"
        )

    if args.dry_run:
        print("Dry-run: staging_metadata.csv not modified.")
        return 0

    write_provenance(metadata, updated)
    print(f"Wrote {metadata}")
    print("Run scripts/promote_staging.py after reviewing bulk approval counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
