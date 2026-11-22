"""Optional OpenCV overlay for detections, direction zones, and HUD."""

from __future__ import annotations

import cv2
import numpy as np

from src.monitoring.performance import PerformanceStats
from src.tracking.tracked_detection import TrackedDetection
from src.utils.config_loader import VisualizationConfig
from src.vision.detection import Detection
from src.vision.direction import Direction

_DIRECTION_COLORS: dict[Direction, tuple[int, int, int]] = {
    Direction.LEFT: (80, 180, 255),
    Direction.CENTER: (80, 220, 80),
    Direction.RIGHT: (255, 180, 80),
}
_ZONE_COLOR = (180, 180, 180)
_HUD_COLOR = (240, 240, 240)
_TEXT_SCALE = 0.55
_TEXT_THICKNESS = 1


class DetectionRenderer:
    """Draw bounding boxes and HUD onto a copy of the original frame."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def draw(
        self,
        frame: np.ndarray,
        detections: list[Detection] | list[TrackedDetection],
        stats: PerformanceStats | None = None,
    ) -> np.ndarray:
        """Return an annotated BGR copy. Does not modify ``frame``."""
        canvas = frame.copy()
        height, width = canvas.shape[:2]

        if self.config.show_direction_zones:
            self._draw_zones(canvas, width, height)

        for detection in detections:
            self._draw_detection(canvas, detection)

        if stats is not None:
            self._draw_hud(canvas, stats)

        return canvas

    def _draw_zones(self, canvas: np.ndarray, width: int, height: int) -> None:
        left_x = int(round(width / 3.0))
        right_x = int(round((2.0 * width) / 3.0))
        cv2.line(canvas, (left_x, 0), (left_x, height), _ZONE_COLOR, 1)
        cv2.line(canvas, (right_x, 0), (right_x, height), _ZONE_COLOR, 1)
        self._put_label(canvas, "LEFT", (8, 22), _ZONE_COLOR)
        self._put_label(canvas, "CENTER", (left_x + 8, 22), _ZONE_COLOR)
        self._put_label(canvas, "RIGHT", (right_x + 8, 22), _ZONE_COLOR)

    def _draw_detection(
        self,
        canvas: np.ndarray,
        detection: Detection | TrackedDetection,
    ) -> None:
        direction = (
            detection.display_direction
            if isinstance(detection, TrackedDetection)
            else detection.direction
        )
        confidence = (
            detection.display_confidence
            if isinstance(detection, TrackedDetection)
            else detection.confidence
        )
        color = _DIRECTION_COLORS.get(direction, (0, 255, 255))
        cv2.rectangle(
            canvas,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )
        if isinstance(detection, TrackedDetection):
            label = (
                f"[ID {detection.track_id}] {detection.class_name} "
                f"{confidence:.2f} {direction.value}"
            )
        else:
            label = f"{detection.class_name} {confidence:.2f} {direction.value}"
        self._put_label(canvas, label, (detection.x1, max(detection.y1 - 8, 16)), color)

    def _draw_hud(self, canvas: np.ndarray, stats: PerformanceStats) -> None:
        lines: list[str] = []
        if self.config.show_fps:
            lines.append(f"FPS: {stats.fps:.1f}")
        if self.config.show_latency:
            lines.append(f"Inference: {stats.inference_ms:.1f} ms")
            if stats.tracking_ms is not None:
                lines.append(f"Tracking: {stats.tracking_ms:.1f} ms")
            lines.append(f"Frame: {stats.frame_ms:.1f} ms")
        y = 46
        for line in lines:
            self._put_label(canvas, line, (8, y), _HUD_COLOR)
            y += 20

    def _put_label(
        self,
        canvas: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        cv2.putText(
            canvas,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            _TEXT_SCALE,
            color,
            _TEXT_THICKNESS,
            cv2.LINE_AA,
        )
