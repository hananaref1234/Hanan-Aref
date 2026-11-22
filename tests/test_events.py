"""Unit tests for duplicate-event suppression (no camera, no YOLO)."""

from __future__ import annotations

from src.tracking.event_suppressor import EventSuppressor, TrackEventKind
from src.tracking.tracked_detection import TrackedDetection
from src.utils.config_loader import EventsConfig
from src.vision.direction import Direction


def _events_config(**overrides: object) -> EventsConfig:
    values = {
        "suppress_duplicates": True,
        "duplicate_cooldown_seconds": 3.0,
        "emit_direction_changes": True,
        "log_lost_tracks": True,
    }
    values.update(overrides)
    return EventsConfig(**values)  # type: ignore[arg-type]


def _track(
    track_id: int = 4,
    direction: Direction = Direction.CENTER,
    confidence: float = 0.93,
    timestamp: float = 1.0,
    confirmed: bool = True,
    class_name: str = "person",
) -> TrackedDetection:
    return TrackedDetection(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        x1=100,
        y1=80,
        x2=220,
        y2=400,
        center_x=160,
        center_y=240,
        direction=direction,
        timestamp=timestamp,
        smoothed_confidence=confidence,
        stable_direction=direction,
        is_confirmed=confirmed,
    )


def test_new_object_emits_one_event() -> None:
    suppressor = EventSuppressor(_events_config())
    events = suppressor.update([_track()], now=10.0)
    assert len(events) == 1
    assert events[0].kind is TrackEventKind.NEW
    assert events[0].track_id == 4
    assert events[0].direction is Direction.CENTER


def test_unconfirmed_object_emits_nothing() -> None:
    suppressor = EventSuppressor(_events_config())
    events = suppressor.update([_track(confirmed=False)], now=10.0)
    assert events == []


def test_unchanged_object_emits_no_duplicate() -> None:
    suppressor = EventSuppressor(_events_config())
    track = _track()
    first = suppressor.update([track], now=10.0)
    later = [suppressor.update([track], now=10.0 + offset) for offset in range(1, 20)]
    assert [event.kind for event in first] == [TrackEventKind.NEW]
    assert all(events == [] for events in later)


def test_direction_change_emits_update() -> None:
    suppressor = EventSuppressor(_events_config())
    suppressor.update([_track(direction=Direction.RIGHT)], now=10.0)
    events = suppressor.update([_track(direction=Direction.CENTER)], now=10.1)
    assert len(events) == 1
    assert events[0].kind is TrackEventKind.UPDATED
    assert events[0].previous_direction is Direction.RIGHT
    assert events[0].direction is Direction.CENTER
    assert "RIGHT->CENTER" in events[0].log_line()


def test_lost_track_emits_once() -> None:
    suppressor = EventSuppressor(_events_config())
    track = _track()
    suppressor.update([track], now=10.0)
    lost_first = suppressor.update([], lost_tracks=[track], now=11.0)
    lost_again = suppressor.update([], lost_tracks=[track], now=12.0)
    assert [event.kind for event in lost_first] == [TrackEventKind.LOST]
    assert lost_first[0].log_line() == "[TRACK_LOST] id=4 class=person"
    assert lost_again == []


def test_reappearing_expired_object_emits_new_event() -> None:
    suppressor = EventSuppressor(_events_config())
    track = _track()
    suppressor.update([track], now=10.0)
    suppressor.update([], lost_tracks=[track], now=16.0)
    events = suppressor.update([track], now=17.0)
    assert [event.kind for event in events] == [TrackEventKind.NEW]


def test_cooldown_does_not_reannounce_persistent_object() -> None:
    suppressor = EventSuppressor(_events_config(duplicate_cooldown_seconds=0.5))
    track = _track()
    suppressor.update([track], now=10.0)
    after_cooldown = suppressor.update([track], now=20.0)
    assert after_cooldown == []
