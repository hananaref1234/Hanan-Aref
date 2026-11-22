"""Build the six-class Exp2 training view. Does not modify the canonical dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.class_map import exp2_six_class_class_map, write_class_map  # noqa: E402
from src.dataset.split_report import count_split_classes, format_balance_report  # noqa: E402
from src.dataset.training_view import (  # noqa: E402
    CANONICAL_DATASET,
    EXP2_TRAINING_VIEW,
    build_training_view,
)
from src.utils.paths import project_root, resolve_from_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data/training/li_stick_v02_exp2/ from the canonical dataset."
    )
    parser.add_argument("--source", type=Path, default=CANONICAL_DATASET)
    parser.add_argument("--dest", type=Path, default=EXP2_TRAINING_VIEW)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source if args.source.is_absolute() else resolve_from_root(args.source)
    dest = args.dest if args.dest.is_absolute() else resolve_from_root(args.dest)
    class_map = exp2_six_class_class_map()
    stats = build_training_view(source, dest, class_map)
    sidecar = project_root() / "models" / "li_stick_navigation_v0.2_exp2.class_map.yaml"
    write_class_map(sidecar, class_map)
    counts = count_split_classes(dest, class_map=class_map)
    print(f"Training view written to {stats.dest_root}")
    print(f"Class map written to {sidecar}")
    print(f"Images: {stats.images}")
    print(f"Annotations: {stats.annotations}")
    print(f"Skipped unmapped boxes: {stats.skipped_unmapped}")
    print(format_balance_report(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
