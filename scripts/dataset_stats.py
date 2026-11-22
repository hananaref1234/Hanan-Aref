"""Print class-balance statistics for the Li-Stick navigation dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.stats import compute_dataset_stats  # noqa: E402
from src.utils.paths import resolve_from_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Li-Stick dataset class balance.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/li_stick_navigation"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset if args.dataset.is_absolute() else resolve_from_root(args.dataset)
    print(compute_dataset_stats(dataset_root).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
