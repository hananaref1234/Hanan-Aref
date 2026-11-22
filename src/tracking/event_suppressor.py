"""State-based navigation-event generation with duplicate suppression."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import time

from src.tracking.tracked_detection import TrackedDetection
from src.utils.config_loader import EventsConfig
from src.vision.direction import Direction


class TrackEventKind(str, Enum):
    NEW = "NEW"
    UPDATED = "UPDATED"
    UNCHANGED = "UNCHANGED"
    LOST = "LOST"


@dataclass(frozen=True)
class TrackEvent:
    """A software navigation event. No speech or vibration is generated."""

    kind: TrackEventKind
    track_id: int
    class_name: str
    class_id: int
    direction: Direction | None
    previous_direction: Direction | None
    confidence: float
    timestamp: float
    track: TrackedDetection | None = None

    def log_line(self) -> str:
        if self.kind is TrackEventKind.NEW:
            return (
                f"[TRACK_NEW] id={self.track_id} class={self.class_name} "
                f"direction={self.direction.value if self.direction else 'UNKNOWN'} "
                f"confidence={self.confidence:.2f}"
            )
        if self.kind is TrackEventKind.UPDATED:
            previous = self.previous_direction.value if self.previous_direction else "?"
            current = self.direction.value if self.direction else "?"
            return (
                f"[TRACK_UPDATE] id={self.track_id} class={self.class_name} "
                f"direction={previous}->{current} "
                f"confidence={self.confidence:.2f}"
            )
        if self.kind is TrackEventKind.LOST:
            return f"[TRACK_LOST] id={self.track_id} class={self.class_name}"
        return (
            f"[TRACK_UNCHANGED] id={self.track_id} class={self.class_name} "
            f"direction={self.direction.value if self.direction else 'UNKNOWN'}"
        )


@dataclass
class _EventState:
    direction: Direction | None
    last_event_kind: TrackEventKind
    last_event_time: float
    lost_emitted: bool = False


class EventSuppressor:
    """Emit NEW / UPDATED / LOST once per meaningful track-state change.

    Persistent unchanged objects stay silent even after
    ``duplicate_cooldown_seconds``. Cooldown only blocks an unjustified
    repeat of the same event; direction changes and true re-acquisitions
    still emit.
    """

    def __init__(self, config: EventsConfig) -> None:
        self.config = config
        self._state: dict[int, _EventState] = {}

    def update(
        self,
        tracks: list[TrackedDetection],
        lost_tracks: list[TrackedDetection] | None = None,
        now: float | None = None,
    ) -> list[TrackEvent]:
        timestamp = time() if now is None else now
        events: list[TrackEvent] = []

        if self.config.suppress_duplicates:
            events.extend(self._events_for_visible(tracks, timestamp))
            events.extend(self._events_for_lost(lost_tracks or [], timestamp))
        else:
            events.extend(
                self._event_from_track(track, TrackEventKind.NEW, timestamp, None)
                for track in tracks
                if track.is_confirmed
            )

        return events

    def _events_for_visible(
        self,
        tracks: list[TrackedDetection],
        timestamp: float,
    ) -> list[TrackEvent]:
        events: list[TrackEvent] = []
        for track in tracks:
            if not track.is_confirmed:
                continue
            direction = track.display_direction
            state = self._state.get(track.track_id)
            if state is None or state.lost_emitted:
                event = self._event_from_track(track, TrackEventKind.NEW, timestamp, None)
                if self._should_emit(state, event, timestamp):
                    events.append(event)
                    self._state[track.track_id] = _EventState(
                        direction=direction,
                        last_event_kind=TrackEventKind.NEW,
                        last_event_time=timestamp,
                        lost_emitted=False,
                    )
                continue

            if (
                self.config.emit_direction_changes
                and state.direction is not None
                and direction is not state.direction
            ):
                event = self._event_from_track(
                    track,
                    TrackEventKind.UPDATED,
                    timestamp,
                    state.direction,
                )
                if self._should_emit(state, event, timestamp):
                    events.append(event)
                    state.direction = direction
                    state.last_event_kind = TrackEventKind.UPDATED
                    state.last_event_time = timestamp
                continue

            # Unchanged: keep internal state, do not emit.
            state.direction = direction

        return events

    def _events_for_lost(
        self,
        lost_tracks: list[TrackedDetection],
        timestamp: float,
    ) -> list[TrackEvent]:
        events: list[TrackEvent] = []
        for track in lost_tracks:
            state = self._state.get(track.track_id)
            if state is None or state.lost_emitted:
                continue
            event = TrackEvent(
                kind=TrackEventKind.LOST,
                track_id=track.track_id,
                class_name=track.class_name,
                class_id=track.class_id,
                direction=track.display_direction,
                previous_direction=state.direction,
                confidence=track.display_confidence,
                timestamp=timestamp,
                track=track,
            )
            events.append(event)
            state.lost_emitted = True
            state.last_event_kind = TrackEventKind.LOST
            state.last_event_time = timestamp
        return events

    def _should_emit(
        self,
        state: _EventState | None,
        event: TrackEvent,
        timestamp: float,
    ) -> bool:
        if state is None or state.lost_emitted:
            return True
        if event.kind is TrackEventKind.UPDATED:
            return True
        elapsed = timestamp - state.last_event_time
        if (
            event.kind is state.last_event_kind
            and event.direction is state.direction
            and elapsed < self.config.duplicate_cooldown_seconds
        ):
            return False
        return True

    def _event_from_track(
        self,
        track: TrackedDetection,
        kind: TrackEventKind,
        timestamp: float,
        previous_direction: Direction | None,
    ) -> TrackEvent:
        return TrackEvent(
            kind=kind,
            track_id=track.track_id,
            class_name=track.class_name,
            class_id=track.class_id,
            direction=track.display_direction,
            previous_direction=previous_direction,
            confidence=track.display_confidence,
            timestamp=timestamp,
            track=track,
        )
