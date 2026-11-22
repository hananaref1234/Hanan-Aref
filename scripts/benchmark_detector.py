"""Benchmark YOLO inference latency for the configured Phase 1 model."""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config_loader import ConfigError, load_app_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.vision.detector import DetectorError, ObjectDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Li-Stick Phase 1 detector inference latency."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=300, help="Timed frames after warmup.")
    parser.add_argument("--warmup", type=int, default=10, help="Uncounted warmup frames.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image used for every iteration. A synthetic frame is used otherwise.",
    )
    return parser.parse_args()


def load_frame(path: Path | None, width: int, height: int) -> np.ndarray:
    if path is None:
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark image not found: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Could not read benchmark image {path}.")
    return frame


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "not installed"
    return getattr(module, "__version__", "unknown")


def main() -> int:
    args = parse_args()
    if args.frames <= 0 or args.warmup < 0:
        print("Error: --frames must be > 0 and --warmup must be >= 0.", file=sys.stderr)
        return 1

    try:
        config = load_app_config(args.config, args.classes)
        setup_logging(config.logging.level)
        frame = load_frame(args.image, config.camera.width, config.camera.height)
        detector = ObjectDetector(
            model_path=config.detector.model_path,
            confidence_threshold=config.detector.confidence_threshold,
            allowed_classes=config.allowed_classes,
            image_size=config.detector.image_size,
        )

        print(f"Warming up for {args.warmup} frames...")
        for _ in range(args.warmup):
            detector.detect(frame)

        latencies: list[float] = []
        print(f"Timing {args.frames} frames...")
        for _ in range(args.frames):
            _, inference_ms = detector.detect(frame)
            latencies.append(inference_ms)

        average = statistics.fmean(latencies)
        median = statistics.median(latencies)
        fps = 1000.0 / average if average > 0 else 0.0

        print()
        print("Li-Stick Detector Benchmark")
        print("---------------------------")
        print(f"Model: {config.detector.model_path}")
        print(f"Image size: {config.detector.image_size}")
        print(f"Frames: {args.frames}")
        print()
        print(f"Average inference: {average:.1f} ms")
        print(f"Median inference: {median:.1f} ms")
        print(f"Approx inference FPS: {fps:.1f}")
        print()
        print("Environment")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Platform: {platform.platform()}")
        print(f"  Machine: {platform.machine()}")
        print(f"  ultralytics: {package_version('ultralytics')}")
        print(f"  opencv-python: {package_version('cv2')}")
        print(f"  numpy: {package_version('numpy')}")
        print()
        print("These timings describe this computer only. They are not Raspberry Pi 5 results.")

    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 0
    except (ConfigError, DetectorError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
