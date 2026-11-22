"""Run v0.3-combined held-out eval plus Exp2b/generic comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.compare_models import compare_custom_vs_exp2b, compare_generic_vs_pretrained
from src.training.config import load_training_config
from src.training.evaluator import evaluate_custom_detector
from src.utils.logging_setup import setup_logging
from src.utils.paths import project_root, resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/training_v03_combined.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/li_stick_navigation_v0.3_combined.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation_v03_combined"))
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def main() -> int:
    setup_logging("INFO")
    args = parse_args()
    training = load_training_config(args.config)
    model_path = args.model if args.model.is_absolute() else resolve_from_root(args.model)
    output_dir = args.output if args.output.is_absolute() else resolve_from_root(args.output)
    evaluate_custom_detector(
        model_path=model_path,
        dataset_yaml=training.dataset_yaml,
        output_dir=output_dir,
        image_size=training.image_size,
        split=args.split,
    )
    from ultralytics import YOLO

    combined = YOLO(str(model_path))
    pretrained = YOLO("yolo26n.pt")
    compare_generic_vs_pretrained(
        combined,
        pretrained,
        training.dataset_yaml.parent,
        output_dir / "generic_vs_pretrained.json",
        split=args.split,
    )
    exp2b = project_root() / "data" / "training" / "li_stick_v02_exp2b"
    if exp2b.exists():
        compare_custom_vs_exp2b(
            combined,
            exp2b,
            output_dir / "custom_vs_exp2b.json",
            split=args.split,
        )
    print(f"Evaluation written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
