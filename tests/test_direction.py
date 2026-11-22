"""Unit tests for LEFT / CENTER / RIGHT boundary behavior."""

from __future__ import annotations

import pytest

from src.vision.direction import Direction, compute_direction


@pytest.mark.parametrize(
    ("center_x", "frame_width", "expected"),
    [
        (0, 900, Direction.LEFT),
        (1, 900, Direction.LEFT),
        (299, 900, Direction.LEFT),
        (299.9, 900, Direction.LEFT),
        (300, 900, Direction.CENTER),
        (450, 900, Direction.CENTER),
        (599, 900, Direction.CENTER),
        (599.9, 900, Direction.CENTER),
        (600, 900, Direction.RIGHT),
        (899, 900, Direction.RIGHT),
        (900, 900, Direction.RIGHT),
        (0, 640, Direction.LEFT),
        (213, 640, Direction.LEFT),
        (640 / 3, 640, Direction.CENTER),
        (320, 640, Direction.CENTER),
        ((2 * 640) / 3, 640, Direction.RIGHT),
        (639, 640, Direction.RIGHT),
        (0, 1280, Direction.LEFT),
        (1280 / 3, 1280, Direction.CENTER),
        (640, 1280, Direction.CENTER),
        ((2 * 1280) / 3, 1280, Direction.RIGHT),
        (1279, 1280, Direction.RIGHT),
        (0, 1920, Direction.LEFT),
        (1920 / 3, 1920, Direction.CENTER),
        ((2 * 1920) / 3, 1920, Direction.RIGHT),
        (1919, 1920, Direction.RIGHT),
        (0, 100, Direction.LEFT),
        (100 / 3, 100, Direction.CENTER),
        ((2 * 100) / 3, 100, Direction.RIGHT),
        (99, 100, Direction.RIGHT),
    ],
)
def test_compute_direction_boundaries(
    center_x: float,
    frame_width: int,
    expected: Direction,
) -> None:
    assert compute_direction(center_x, frame_width) is expected


def test_exact_first_third_is_center() -> None:
    assert compute_direction(300, 900) is Direction.CENTER


def test_exact_second_third_is_right() -> None:
    assert compute_direction(600, 900) is Direction.RIGHT


def test_just_inside_left_band() -> None:
    assert compute_direction(299.999, 900) is Direction.LEFT


def test_invalid_frame_width() -> None:
    with pytest.raises(ValueError, match="frame_width"):
        compute_direction(10, 0)
    with pytest.raises(ValueError, match="frame_width"):
        compute_direction(10, -640)
