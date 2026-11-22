"""Train a custom Li-Stick detector. Refuses to run without a labeled dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.config import TrainingConfigError, load_training_config  # noqa: E402
from src.training.trainer import TrainingNotReadyError, train_custom_detector  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Li-Stick custom navigation detector (transfer learning)."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config/training.yaml.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto, cpu, cuda, or a device index.")
    parser.add_argument("--name", type=str, default=None, help="Experiment name.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from runs/<project>/<name>/weights/last.pt.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging("INFO")
    args = parse_args()
    try:
        config = load_training_config(args.config)
        if args.epochs is not None or args.batch_size is not None or args.image_size is not None or args.device or args.name:
            from dataclasses import replace

            config = replace(
                config,
                epochs=args.epochs if args.epochs is not None else config.epochs,
                batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
                image_size=args.image_size if args.image_size is not None else config.image_size,
                device=args.device if args.device is not None else config.device,
                name=args.name if args.name is not None else config.name,
            )
        output = train_custom_detector(config, resume=args.resume)
        print(f"Training finished. Output: {output}")
        return 0
    except (TrainingConfigError, TrainingNotReadyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
