"""Download and filter the official COCO 2017 chair/generic subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.coco_generic_subset import DEFAULT_RAW_DIR, prepare_coco_chair_subset
from src.utils.paths import resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--chair-cap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest = args.dest if args.dest.is_absolute() else resolve_from_root(args.dest)
    stats = prepare_coco_chair_subset(
        dest,
        chair_cap=args.chair_cap,
        seed=args.seed,
        download=not args.no_download,
    )
    print(f"Wrote {stats.images} images / {stats.annotations} boxes to {stats.raw_dir}")
    print("Per class:", stats.per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
