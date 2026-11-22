"""Copy reviewed staging samples into data/li_stick_navigation. Does not train."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers.promote import PromotionError, promote_staging
from src.dataset.importers.provenance import default_metadata_path
from src.utils.paths import project_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        default=str(project_root() / "data" / "staging"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(project_root() / "data" / "li_stick_navigation"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = default_metadata_path(Path(args.staging_root))
    try:
        summary = promote_staging(
            metadata,
            Path(args.dataset_root),
            dry_run=args.dry_run,
        )
    except PromotionError as exc:
        print(exc)
        return 1
    print(
        f"Promoted {summary['promoted_boxes']} boxes in {summary['images']} images"
        f"{' (dry-run)' if args.dry_run else ''}."
    )
    print("Do not train until the assembled dataset is reviewed and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
