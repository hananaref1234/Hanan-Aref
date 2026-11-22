"""Evaluate a custom Li-Stick detector. Requires a trained checkpoint and labeled val set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.config import load_training_config  # noqa: E402
from src.training.evaluator import evaluate_custom_detector  # noqa: E402
from src.training.trainer import TrainingNotReadyError  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.utils.paths import resolve_from_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a custom Li-Stick YOLO checkpoint.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None, help="Path to a trained .pt checkpoint.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Held-out split to evaluate. Default: test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory for metrics, safety report, and TP/FP/FN images.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging("INFO")
    args = parse_args()
    try:
        training = load_training_config(args.config)
        model_path = args.model
        if model_path is None:
            model_path = training.project / training.name / "weights" / "best.pt"
        elif not model_path.is_absolute():
            model_path = resolve_from_root(model_path)
        output_dir = args.output if args.output.is_absolute() else resolve_from_root(args.output)
        evaluate_custom_detector(
            model_path=model_path,
            dataset_yaml=training.dataset_yaml,
            output_dir=output_dir,
            image_size=training.image_size,
            split=args.split,
        )
        print(f"Evaluation written to {output_dir}")
        return 0
    except TrainingNotReadyError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
