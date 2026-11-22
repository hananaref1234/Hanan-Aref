"""Minimal inference example for hardware-team integration.

Demonstrates:
  1. Loading the v0.3-combined config and class filter
  2. Building an ObjectDetector
  3. Running inference on a single image file or a camera frame
  4. Printing structured Detection output

This script does NOT implement its own detector. It calls the same
ObjectDetector used by the full pipeline.

Usage:
    # Run on an image file:
    python examples/run_detector.py --image path/to/image.jpg

    # Run a quick test with a black dummy frame (no camera needed):
    python examples/run_detector.py --dummy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.utils.config_loader import load_app_config
from src.utils.logging_setup import setup_logging
from src.vision.detector import ObjectDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to an image file.")
    group.add_argument("--dummy", action="store_true", help="Use a synthetic black frame.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config_v03_combined.yaml"),
        help="Config YAML (default: config/config_v03_combined.yaml).",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("config/classes_v03_combined.yaml"),
        help="Class filter YAML (default: config/classes_v03_combined.yaml).",
    )
    return parser.parse_args()


def load_frame(args: argparse.Namespace) -> "np.ndarray":
    if args.dummy:
        print("Using dummy 640×640 black frame (no detections expected).")
        return np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        import cv2
    except ImportError as exc:
        sys.exit(f"opencv-python is required to read image files: {exc}")

    frame = cv2.imread(str(args.image))
    if frame is None:
        sys.exit(f"Could not read image: {args.image}")
    print(f"Loaded image: {args.image}  shape={frame.shape}")
    return frame


def main() -> int:
    setup_logging("INFO")
    args = parse_args()

    config = load_app_config(args.config, args.classes)

    detector = ObjectDetector(
        model_path=config.detector.model_path,
        confidence_threshold=config.detector.confidence_threshold,
        allowed_classes=config.allowed_classes,
        image_size=config.detector.image_size,
    )

    frame = load_frame(args)
    detections, inference_ms = detector.detect(frame)

    print(f"\nInference time: {inference_ms:.1f} ms")
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(json.dumps(det.to_dict(), indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
