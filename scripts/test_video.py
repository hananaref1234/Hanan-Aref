"""Run the Phase 1 detector on a recorded video file."""

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

from src.camera.base_camera import CameraError  # noqa: E402
from src.camera.opencv_camera import OpenCVCamera  # noqa: E402
from src.monitoring.performance import PerformanceMonitor  # noqa: E402
from src.tracking.event_suppressor import EventSuppressor, TrackEventKind  # noqa: E402
from src.tracking.stabilization import TemporalStabilizer  # noqa: E402
from src.tracking.tracker import TrackerError, create_tracker  # noqa: E402
from src.utils.config_loader import ConfigError, load_app_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.vision.detector import DetectorError, ObjectDetector  # noqa: E402
from src.visualization.renderer import DetectionRenderer  # noqa: E402

logger = logging.getLogger("li_stick.test_video")
_WINDOW_NAME = "Li-Stick test video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Li-Stick Phase 1 detection on a video file."
    )
    parser.add_argument("video", type=Path, help="Path to a video file such as sample.mp4.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the preview window even if visualization.enabled is true.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config(args.config, args.classes)
        setup_logging(config.logging.level)

        detector = ObjectDetector(
            model_path=config.detector.model_path,
            confidence_threshold=config.detector.confidence_threshold,
            allowed_classes=config.allowed_classes,
            image_size=config.detector.image_size,
        )
        camera = OpenCVCamera(
            device=str(args.video),
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
        )
        renderer = (
            DetectionRenderer(config.visualization)
            if config.visualization.enabled and not args.no_display
            else None
        )
        monitor = PerformanceMonitor()
        tracker = create_tracker(config.tracking) if config.tracking.enabled else None
        stabilizer = TemporalStabilizer(config.tracking) if config.tracking.enabled else None
        suppressor = EventSuppressor(config.events) if config.tracking.enabled else None

        camera.open()
        try:
            while True:
                frame_started = time.perf_counter()
                ok, frame = camera.read()
                if not ok or frame is None:
                    logger.info("End of video file.")
                    break

                detections, inference_ms = detector.detect(frame, timestamp=time.time())
                for detection in detections:
                    logger.debug(detection.log_line())

                overlay_objects = detections
                tracking_ms: float | None = None
                if tracker is not None and stabilizer is not None and suppressor is not None:
                    track_started = time.perf_counter()
                    tracked = tracker.update(detections, frame)
                    tracking_ms = (time.perf_counter() - track_started) * 1000.0
                    stable, lost = stabilizer.update(tracked)
                    events = suppressor.update(stable, lost_tracks=lost)
                    overlay_objects = stable
                    for event in events:
                        if event.kind is TrackEventKind.LOST and not config.events.log_lost_tracks:
                            continue
                        logger.info(event.log_line())

                stats = monitor.record(
                    inference_ms,
                    (time.perf_counter() - frame_started) * 1000.0,
                    tracking_ms=tracking_ms,
                )

                if renderer is None:
                    continue
                annotated = renderer.draw(frame, overlay_objects, stats)
                cv2.imshow(_WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            camera.close()
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        cv2.destroyAllWindows()
        return 0
    except (ConfigError, DetectorError, CameraError, TrackerError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        cv2.destroyAllWindows()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
