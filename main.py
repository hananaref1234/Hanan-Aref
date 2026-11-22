"""Li-Stick live-camera loop: Phase 1 detection plus Phase 2 tracking.

Phase 2 adds object tracking, temporal stabilization, and duplicate-event
suppression. ESP32, serial, sensor fusion, speech, and vibration are still
not implemented.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from src.camera.base_camera import CameraError
from src.camera.factory import create_camera
from src.monitoring.performance import PerformanceMonitor
from src.tracking.event_suppressor import EventSuppressor, TrackEventKind
from src.tracking.stabilization import TemporalStabilizer
from src.tracking.tracker import TrackerError, create_tracker
from src.utils.config_loader import AppConfig, ConfigError, load_app_config
from src.utils.logging_setup import setup_logging
from src.vision.detector import DetectorError, ObjectDetector
from src.visualization.renderer import DetectionRenderer

logger = logging.getLogger("li_stick")
_WINDOW_NAME = "Li-Stick Phase 2"
_MAX_LIVE_FAILURES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Li-Stick live object detection and tracking (camera only)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (defaults to config/config.yaml).",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=None,
        help="Path to classes.yaml (defaults to config/classes.yaml).",
    )
    return parser.parse_args()


def build_detector(config: AppConfig) -> ObjectDetector:
    return ObjectDetector(
        model_path=config.detector.model_path,
        confidence_threshold=config.detector.confidence_threshold,
        allowed_classes=config.allowed_classes,
        image_size=config.detector.image_size,
    )


def run_loop(config: AppConfig, detector: ObjectDetector) -> None:
    camera = create_camera(config.camera)
    monitor = PerformanceMonitor(window_size=30)
    renderer = DetectionRenderer(config.visualization) if config.visualization.enabled else None
    tracker = create_tracker(config.tracking) if config.tracking.enabled else None
    stabilizer = TemporalStabilizer(config.tracking) if config.tracking.enabled else None
    suppressor = EventSuppressor(config.events) if config.tracking.enabled else None
    consecutive_failures = 0

    camera.open()
    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = camera.read()
            if not ok or frame is None:
                consecutive_failures += 1
                is_file = getattr(camera, "is_file_source", False)
                if is_file:
                    logger.info("End of video stream.")
                    break
                logger.warning(
                    "Frame capture failed (%s/%s). Check the camera connection.",
                    consecutive_failures,
                    _MAX_LIVE_FAILURES,
                )
                if consecutive_failures >= _MAX_LIVE_FAILURES:
                    raise CameraError(
                        "Camera stopped delivering frames. "
                        "Reconnect the webcam or switch camera.device in config/config.yaml."
                    )
                continue

            consecutive_failures = 0
            timestamp = time.time()
            detections, inference_ms = detector.detect(frame, timestamp=timestamp)
            for detection in detections:
                logger.debug(detection.log_line())

            overlay_objects = detections
            tracking_ms: float | None = None
            if tracker is not None and stabilizer is not None and suppressor is not None:
                track_started = time.perf_counter()
                tracked = tracker.update(detections, frame)
                tracking_ms = (time.perf_counter() - track_started) * 1000.0
                stable, lost = stabilizer.update(tracked)
                events = suppressor.update(stable, lost_tracks=lost, now=timestamp)
                overlay_objects = stable
                for event in events:
                    if event.kind is TrackEventKind.LOST and not config.events.log_lost_tracks:
                        logger.debug(event.log_line())
                        continue
                    logger.info(event.log_line())

            frame_ms = (time.perf_counter() - frame_started) * 1000.0
            stats = monitor.record(inference_ms, frame_ms, tracking_ms=tracking_ms)
            logger.debug(
                "Performance inference=%.1fms tracking=%s frame=%.1fms fps=%.1f",
                stats.inference_ms,
                f"{stats.tracking_ms:.1f}ms" if stats.tracking_ms is not None else "n/a",
                stats.frame_ms,
                stats.fps,
            )

            if renderer is None:
                continue

            annotated = renderer.draw(frame, overlay_objects, stats)
            try:
                cv2.imshow(_WINDOW_NAME, annotated)
            except cv2.error as exc:
                logger.error(
                    "Could not open a preview window (%s). "
                    "Set visualization.enabled to false for headless Raspberry Pi use.",
                    exc,
                )
                renderer = None
                continue

            if renderer is not None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("Exit requested from preview window.")
                    break
    finally:
        camera.close()
        if renderer is not None:
            cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config(args.config, args.classes)
        setup_logging(config.logging.level)
        logger.info("Starting Li-Stick Phase 2 detector + tracker")
        logger.info(
            "Model=%s image_size=%s confidence>=%.2f class_filter=%s tracking=%s",
            config.detector.model_path,
            config.detector.image_size,
            config.detector.confidence_threshold,
            config.detector.use_class_filter,
            config.tracking.enabled,
        )
        detector = build_detector(config)
        run_loop(config, detector)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")
        return 0
    except (ConfigError, DetectorError, CameraError, TrackerError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
