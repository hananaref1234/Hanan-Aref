"""Tracker association tests using synthetic detections (no webcam)."""

from __future__ import annotations

import numpy as np

from src.tracking.tracked_detection import TrackedDetection
from src.tracking.tracker import ObjectTracker
from src.utils.config_loader import TrackingConfig
from src.vision.detection import Detection
from src.vision.direction import Direction


def _config(**overrides: object) -> TrackingConfig:
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


def _detection(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.94,
    timestamp: float = 1.0,
    frame_width: int = 1280,
) -> Detection:
    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))
    if center_x < frame_width / 3:
        direction = Direction.LEFT
    elif center_x < (2 * frame_width) / 3:
        direction = Direction.CENTER
    else:
        direction = Direction.RIGHT
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=center_x,
        center_y=center_y,
        direction=direction,
        timestamp=timestamp,
    )


def _frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_tracked_detection_to_dict() -> None:
    payload = TrackedDetection(
        track_id=7,
        class_id=0,
        class_name="person",
        confidence=0.94,
        x1=460,
        y1=180,
        x2=1114,
        y2=717,
        center_x=787,
        center_y=448,
        direction=Direction.CENTER,
        timestamp=1760000000.2,
    ).to_dict()
    assert payload["track_id"] == 7
    assert payload["class_name"] == "person"
    assert payload["confidence"] == 0.94
    assert payload["bbox"] == {"x1": 460, "y1": 180, "x2": 1114, "y2": 717}
    assert payload["direction"] == "CENTER"
    assert payload["timestamp"] == 1760000000.2
    assert "track_id" in payload
    assert "risk_level" not in payload


def test_same_object_keeps_track_id() -> None:
    tracker = ObjectTracker(_config())
    frame = _frame()
    first = tracker.update([_detection(200, 100, 360, 500)], frame)
    second = tracker.update([_detection(210, 110, 370, 510)], frame)
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id
    assert first[0].class_name == "person"


def test_second_object_gets_new_track_id() -> None:
    tracker = ObjectTracker(_config())
    frame = _frame()
    tracker.update([_detection(80, 100, 220, 500)], frame)
    tracker.update(
        [
            _detection(90, 110, 230, 510),
            _detection(900, 120, 1100, 520),
        ],
        frame,
    )
    tracked = tracker.update(
        [
            _detection(95, 112, 235, 512),
            _detection(905, 125, 1105, 525),
        ],
        frame,
    )
    ids = {item.track_id for item in tracked}
    assert len(tracked) == 2
    assert len(ids) == 2


def test_short_gap_can_retain_track_id() -> None:
    tracker = ObjectTracker(_config(max_missing_frames=5))
    frame = _frame()
    first = tracker.update([_detection(200, 100, 360, 500)], frame)
    tracker.update([_detection(205, 105, 365, 505)], frame)
    original_id = first[0].track_id
    tracker.update([], frame)
    tracker.update([], frame)
    recovered = tracker.update([_detection(210, 110, 370, 510)], frame)
    assert len(recovered) == 1
    assert recovered[0].track_id == original_id
