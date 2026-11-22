"""Internal detection representation used by the rest of Li-Stick."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.vision.direction import Direction


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in original-frame pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class Detection:
    """One filtered object detection in Li-Stick's internal format.

    Future tracking / fusion fields (track_id, risk_level, source,
    relevance_score) are omitted rather than filled with placeholders.
    ``depth_m`` is reserved for a later real depth source and stays None.
    Bounding-box size is not a distance.
    """

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
    depth_m: float | None = None

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(self.x1, self.y1, self.x2, self.y2)

    def log_line(self) -> str:
        """Compact console line for accepted detections."""
        return (
            f"[DETECTION] class={self.class_name} "
            f"confidence={self.confidence:.2f} "
            f"direction={self.direction.value} "
            f"bbox=({self.x1},{self.y1},{self.x2},{self.y2})"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable payload for later tracking / serial / fusion."""
        payload: dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "center": {"x": self.center_x, "y": self.center_y},
            "direction": self.direction.value,
            "timestamp": self.timestamp,
        }
        if self.depth_m is not None:
            payload["depth_m"] = self.depth_m
        return payload
