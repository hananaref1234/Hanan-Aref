"""Build a derived stairs/curb training view. Does not modify the canonical dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.class_map import exp1_stairs_curb_class_map  # noqa: E402
from src.dataset.training_view import (  # noqa: E402
    CANONICAL_DATASET,
    EXP1_TRAINING_VIEW,
    build_training_view,
)
from src.utils.paths import resolve_from_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data/training/li_stick_v02_exp1/ from the canonical dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=CANONICAL_DATASET,
        help="Canonical dataset root (never overwritten).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=EXP1_TRAINING_VIEW,
        help="Derived training-view directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source if args.source.is_absolute() else resolve_from_root(args.source)
    dest = args.dest if args.dest.is_absolute() else resolve_from_root(args.dest)
    stats = build_training_view(source, dest, exp1_stairs_curb_class_map())
    print(f"Training view written to {stats.dest_root}")
    print(f"Images: {stats.images}")
    print(f"Annotations: {stats.annotations}")
    print(f"Skipped unmapped boxes: {stats.skipped_unmapped}")
    print(f"Hardlinked images: {stats.linked_images}")
    print(f"Copied images: {stats.copied_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
