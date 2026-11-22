"""LEFT / CENTER / RIGHT localization from a bounding-box center."""

from __future__ import annotations

from enum import Enum


class Direction(str, Enum):
    """Horizontal region of the frame occupied by a detection center."""

    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


def compute_direction(center_x: float, frame_width: int) -> Direction:
    """Map a bounding-box center x-coordinate onto LEFT / CENTER / RIGHT.

    The frame is split into three equal vertical bands:

    - LEFT:   ``center_x < width / 3``
    - CENTER: ``width / 3 <= center_x < 2 * width / 3``
    - RIGHT:  ``center_x >= 2 * width / 3``

    Boundaries are inclusive on the right of each interval except LEFT, so
    an exact third-boundary is deterministic: ``W/3`` is CENTER and
    ``2W/3`` is RIGHT.
    """
    if frame_width <= 0:
        raise ValueError(
            f"frame_width must be > 0 to compute direction, got {frame_width}."
        )

    left_limit = frame_width / 3.0
    right_limit = (2.0 * frame_width) / 3.0

    if center_x < left_limit:
        return Direction.LEFT
    if center_x < right_limit:
        return Direction.CENTER
    return Direction.RIGHT
