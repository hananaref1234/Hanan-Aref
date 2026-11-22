"""Validate the Li-Stick navigation dataset. Does not invent images or labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.validator import validate_dataset  # noqa: E402
from src.utils.paths import resolve_from_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Li-Stick YOLO dataset files.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/li_stick_navigation"),
        help="Dataset root containing dataset.yaml, images/, and labels/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset if args.dataset.is_absolute() else resolve_from_root(args.dataset)
    report = validate_dataset(dataset_root)
    print(report.summary())
    if report.empty:
        return 2
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
