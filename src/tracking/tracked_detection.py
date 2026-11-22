"""Tracked detection representation used after the Phase 1 detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.vision.detection import BoundingBox, Detection
from src.vision.direction import Direction


@dataclass(frozen=True)
class TrackedDetection:
    """One tracked object for a single frame.

    ``confidence`` and ``direction`` are the raw values from this frame.
    ``smoothed_confidence`` and ``stable_direction`` are filled by the
    temporal stabilizer. Future risk/distance fields are omitted rather
    than faked.
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    direction: Direction
    timestamp: float
    smoothed_confidence: float | None = None
    stable_direction: Direction | None = None
    hit_count: int = 1
    missing_count: int = 0
    is_confirmed: bool = False
    depth_m: float | None = None

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(self.x1, self.y1, self.x2, self.y2)

    @property
    def display_confidence(self) -> float:
        if self.smoothed_confidence is None:
            return self.confidence
        return self.smoothed_confidence

    @property
    def display_direction(self) -> Direction:
        if self.stable_direction is None:
            return self.direction
        return self.stable_direction

    def log_line(self) -> str:
        return (
            f"[TRACK] id={self.track_id} class={self.class_name} "
            f"confidence={self.display_confidence:.2f} "
            f"direction={self.display_direction.value} "
            f"bbox=({self.x1},{self.y1},{self.x2},{self.y2})"
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "center": {"x": self.center_x, "y": self.center_y},
            "direction": self.direction.value,
            "timestamp": self.timestamp,
        }
        if self.smoothed_confidence is not None:
            payload["smoothed_confidence"] = self.smoothed_confidence
        if self.stable_direction is not None:
            payload["stable_direction"] = self.stable_direction.value
        payload["is_confirmed"] = self.is_confirmed
        return payload

    @classmethod
    def from_detection(cls, detection: Detection, track_id: int) -> TrackedDetection:
        """Wrap a Phase 1 detection with a track ID. Does not mutate Detection."""
        return cls(
            track_id=track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            x1=detection.x1,
            y1=detection.y1,
            x2=detection.x2,
            y2=detection.y2,
            center_x=detection.center_x,
            center_y=detection.center_y,
            direction=detection.direction,
            timestamp=detection.timestamp,
            depth_m=detection.depth_m,
        )
