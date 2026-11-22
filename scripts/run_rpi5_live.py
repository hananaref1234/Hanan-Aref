"""Raspberry Pi 5 live-camera entrypoint.

Thin wrapper around main.py that defaults to the Pi deployment config and
the v0.3-combined class filter. Reuses the full detection + tracking pipeline
without duplicating any logic.

Usage (on the Pi):
    python scripts/run_rpi5_live.py

Override config or class filter:
    python scripts/run_rpi5_live.py \
        --config config/config_rpi5_combined.yaml \
        --classes config/classes_v03_combined.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import resolve_from_root

_DEFAULT_CONFIG = Path("config/config_rpi5_combined.yaml")
_DEFAULT_CLASSES = Path("config/classes_v03_combined.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Li-Stick v0.3-combined detector on a Raspberry Pi 5 "
            "with Camera Module V2 NoIR."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Main YAML config (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=_DEFAULT_CLASSES,
        help=f"Class-filter YAML (default: {_DEFAULT_CLASSES}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Delegate entirely to main.py's entry point.
    import main as _main_module  # noqa: PLC0415

    sys.argv = [
        "main.py",
        "--config",
        str(resolve_from_root(args.config)),
        "--classes",
        str(resolve_from_root(args.classes)),
    ]
    return _main_module.main()


if __name__ == "__main__":
    raise SystemExit(main())
