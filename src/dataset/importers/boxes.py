"""Bounding-box geometry helpers for importers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def clip(self, image_width: int, image_height: int) -> PixelBox:
        x1 = min(max(self.x1, 0.0), float(image_width))
        y1 = min(max(self.y1, 0.0), float(image_height))
        x2 = min(max(self.x2, 0.0), float(image_width))
        y2 = min(max(self.y2, 0.0), float(image_height))
        return PixelBox(x1, y1, x2, y2)


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def xyxy_to_yolo(
    box: PixelBox,
    image_width: int,
    image_height: int,
    class_id: int,
) -> YoloBox:
    """Convert a pixel xyxy box to a clipped normalized YOLO box."""
    clipped = box.clip(image_width, image_height)
    if clipped.width <= 0 or clipped.height <= 0:
        raise ValueError("Box has no area after clipping.")
    x_center = ((clipped.x1 + clipped.x2) / 2.0) / image_width
    y_center = ((clipped.y1 + clipped.y2) / 2.0) / image_height
    width = clipped.width / image_width
    height = clipped.height / image_height
    return YoloBox(
        class_id=class_id,
        x_center=_clamp01(x_center),
        y_center=_clamp01(y_center),
        width=_clamp01(width),
        height=_clamp01(height),
    )


def yolo_to_xyxy(
    box: YoloBox,
    image_width: int,
    image_height: int,
) -> PixelBox:
    x_center = box.x_center * image_width
    y_center = box.y_center * image_height
    width = box.width * image_width
    height = box.height * image_height
    return PixelBox(
        x1=x_center - width / 2.0,
        y1=y_center - height / 2.0,
        x2=x_center + width / 2.0,
        y2=y_center + height / 2.0,
    )


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
