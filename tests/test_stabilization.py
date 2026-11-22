"""Unit tests for temporal confirmation, confidence EMA, and direction voting."""

from __future__ import annotations

import pytest

from src.tracking.stabilization import (
    TemporalStabilizer,
    smooth_confidence,
    stabilize_direction,
)
from src.tracking.tracked_detection import TrackedDetection
from src.utils.config_loader import TrackingConfig
from src.vision.direction import Direction


def _tracking_config(**overrides: object) -> TrackingConfig:
    values = {
        "enabled": True,
        "tracker": "bytetrack",
        "confirmation_frames": 3,
        "max_missing_frames": 5,
        "confidence_smoothing_alpha": 0.4,
        "direction_history_size": 5,
    }
    values.update(overrides)
    return TrackingConfig(**values)  # type: ignore[arg-type]


def _track(
    track_id: int = 7,
    direction: Direction = Direction.CENTER,
    confidence: float = 0.9,
    timestamp: float = 1.0,
) -> TrackedDetection:
    return TrackedDetection(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=confidence,
        x1=400,
        y1=100,
        x2=700,
        y2=500,
        center_x=550,
        center_y=300,
        direction=direction,
        timestamp=timestamp,
    )


def test_noisy_direction_majority_is_center() -> None:
    sequence = [
        Direction.CENTER,
        Direction.CENTER,
        Direction.RIGHT,
        Direction.CENTER,
        Direction.CENTER,
    ]
    assert stabilize_direction(sequence) is Direction.CENTER


def test_confidence_ema_matches_expected_value() -> None:
    smoothed = smooth_confidence([0.91, 0.93, 0.89, 0.95], alpha=0.4)
    assert smoothed == pytest.approx(0.92408)


def test_stabilizer_confirms_after_consecutive_hits() -> None:
    stabilizer = TemporalStabilizer(_tracking_config(confirmation_frames=3))
    visible, lost = stabilizer.update([_track()])
    assert visible[0].is_confirmed is False
    assert lost == []
    visible, _ = stabilizer.update([_track()])
    assert visible[0].is_confirmed is False
    visible, _ = stabilizer.update([_track()])
    assert visible[0].is_confirmed is True
    assert visible[0].hit_count == 3


def test_stabilizer_smooths_confidence() -> None:
    stabilizer = TemporalStabilizer(_tracking_config(confidence_smoothing_alpha=0.4))
    result = None
    for value in (0.91, 0.93, 0.89, 0.95):
        result, _ = stabilizer.update([_track(confidence=value)])
    assert result is not None
    assert result[0].confidence == pytest.approx(0.95)
    assert result[0].smoothed_confidence == pytest.approx(0.92408)


def test_stabilizer_direction_ignores_one_frame_jitter() -> None:
    stabilizer = TemporalStabilizer(_tracking_config(direction_history_size=5))
    last = None
    for direction in (
        Direction.CENTER,
        Direction.CENTER,
        Direction.RIGHT,
        Direction.CENTER,
        Direction.CENTER,
    ):
        last, _ = stabilizer.update([_track(direction=direction)])
    assert last is not None
    assert last[0].direction is Direction.CENTER
    assert last[0].stable_direction is Direction.CENTER


def test_confirmed_track_is_lost_after_missing_frames() -> None:
    stabilizer = TemporalStabilizer(
        _tracking_config(confirmation_frames=1, max_missing_frames=2)
    )
    stabilizer.update([_track()])
    _, lost = stabilizer.update([])
    assert lost == []
    _, lost = stabilizer.update([])
    assert lost == []
    _, lost = stabilizer.update([])
    assert len(lost) == 1
    assert lost[0].track_id == 7
    _, lost_again = stabilizer.update([])
    assert lost_again == []
