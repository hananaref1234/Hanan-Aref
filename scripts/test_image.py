"""Run the Phase 1 detector once on a still image."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.performance import PerformanceMonitor  # noqa: E402
from src.utils.config_loader import ConfigError, load_app_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.vision.detector import DetectorError, ObjectDetector  # noqa: E402
from src.visualization.renderer import DetectionRenderer  # noqa: E402

logger = logging.getLogger("li_stick.test_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Li-Stick Phase 1 detection on a still image."
    )
    parser.add_argument("image", type=Path, help="Path to an image file.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path for an annotated output image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an annotated preview window (press any key to close).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config(args.config, args.classes)
        setup_logging(config.logging.level)

        if not args.image.exists():
            raise FileNotFoundError(
                f"Image not found: {args.image}. Pass an existing .jpg/.png file."
            )

        frame = cv2.imread(str(args.image))
        if frame is None:
            raise RuntimeError(
                f"Could not read image {args.image}. "
                "The file may be corrupt or an unsupported format."
            )

        detector = ObjectDetector(
            model_path=config.detector.model_path,
            confidence_threshold=config.detector.confidence_threshold,
            allowed_classes=config.allowed_classes,
            image_size=config.detector.image_size,
        )
        started = time.perf_counter()
        detections, inference_ms = detector.detect(frame, timestamp=time.time())
        frame_ms = (time.perf_counter() - started) * 1000.0
        stats = PerformanceMonitor(window_size=1).record(inference_ms, frame_ms)

        if detections:
            for detection in detections:
                line = detection.log_line()
                logger.info(line)
                print(line)
        else:
            print("No detections passed the confidence / class filters.")

        print(
            f"Inference: {stats.inference_ms:.1f} ms  "
            f"Frame: {stats.frame_ms:.1f} ms  "
            f"Detections: {len(detections)}"
        )

        if args.show or args.save is not None:
            annotated = DetectionRenderer(config.visualization).draw(
                frame, detections, stats
            )
            if args.save is not None:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save), annotated):
                    raise RuntimeError(f"Could not write annotated image to {args.save}.")
                print(f"Saved annotated image to {args.save}")
            if args.show:
                cv2.imshow("Li-Stick test image", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("Interrupted by user.")
        cv2.destroyAllWindows()
        return 0
    except (ConfigError, DetectorError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        cv2.destroyAllWindows()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
