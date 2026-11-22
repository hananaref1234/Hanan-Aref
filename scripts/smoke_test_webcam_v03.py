"""Brief webcam smoke test for the v0.3-combined detector.

Does not claim real-world validation. Exits if no camera is available.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.utils.config_loader import load_app_config
from src.utils.paths import project_root
from src.vision.detector import ObjectDetector

CUSTOM = {"pole", "bollard", "stairs", "curb", "barrier", "tree_trunk"}
GENERIC = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "chair"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_app_config(
        project_root() / "config" / "config_v03_combined.yaml",
        project_root() / "config" / "classes_v03_combined.yaml",
    )
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("WEBCAM SMOKE TEST: SKIPPED (no camera)")
        return 0
    detector = ObjectDetector(
        model_path=config.detector.model_path,
        confidence_threshold=config.detector.confidence_threshold,
        allowed_classes=config.allowed_classes,
        image_size=config.detector.image_size,
    )
    started = time.time()
    counts: Counter[str] = Counter()
    frames = 0
    while time.time() - started < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        detections, _ms = detector.detect(frame)
        counts.update(item.class_name for item in detections)
    cap.release()
    custom_seen = sorted(name for name in counts if name in CUSTOM)
    generic_seen = sorted(name for name in counts if name in GENERIC)
    print(f"WEBCAM SMOKE TEST: frames={frames} detections={dict(counts)}")
    print(f"CUSTOM seen: {custom_seen or 'none'}")
    print(f"GENERIC seen: {generic_seen or 'none'}")
    print("This is not real-world validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
