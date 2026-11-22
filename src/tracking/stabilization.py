"""Temporal confirmation, confidence smoothing, and direction majority vote."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace

from src.tracking.tracked_detection import TrackedDetection
from src.utils.config_loader import TrackingConfig
from src.vision.direction import Direction


@dataclass
class _TrackMemory:
    hit_count: int
    consecutive_hits: int
    missing_count: int
    ema_confidence: float
    directions: deque[Direction]
    last_stable_direction: Direction | None
    last_track: TrackedDetection
    was_confirmed: bool


class TemporalStabilizer:
    """Turn noisy per-frame tracks into confirmed, smoothed track states.

    A track is confirmed after ``confirmation_frames`` consecutive hits.
    Once confirmed it stays confirmed until it is missing for more than
    ``max_missing_frames``. Lost unconfirmed tracks are dropped silently.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._memory: dict[int, _TrackMemory] = {}

    def update(
        self,
        tracks: list[TrackedDetection],
    ) -> tuple[list[TrackedDetection], list[TrackedDetection]]:
        """Return ``(visible_tracks, newly_lost_confirmed_tracks)``."""
        current_ids = {track.track_id for track in tracks}
        stabilized: list[TrackedDetection] = []

        for track in tracks:
            memory = self._touch(track)
            enriched = replace(
                track,
                smoothed_confidence=memory.ema_confidence,
                stable_direction=memory.last_stable_direction,
                hit_count=memory.hit_count,
                missing_count=0,
                is_confirmed=memory.was_confirmed,
            )
            memory.last_track = enriched
            stabilized.append(enriched)

        lost: list[TrackedDetection] = []
        for track_id, memory in list(self._memory.items()):
            if track_id in current_ids:
                continue
            memory.missing_count += 1
            memory.consecutive_hits = 0
            if memory.missing_count > self.config.max_missing_frames:
                if memory.was_confirmed:
                    lost.append(
                        replace(
                            memory.last_track,
                            missing_count=memory.missing_count,
                            is_confirmed=True,
                        )
                    )
                del self._memory[track_id]

        return stabilized, lost

    def _touch(self, track: TrackedDetection) -> _TrackMemory:
        memory = self._memory.get(track.track_id)
        if memory is None:
            memory = _TrackMemory(
                hit_count=1,
                consecutive_hits=1,
                missing_count=0,
                ema_confidence=track.confidence,
                directions=deque(
                    [track.direction],
                    maxlen=self.config.direction_history_size,
                ),
                last_stable_direction=track.direction,
                last_track=track,
                was_confirmed=self.config.confirmation_frames <= 1,
            )
            self._memory[track.track_id] = memory
            return memory

        memory.hit_count += 1
        memory.consecutive_hits += 1
        memory.missing_count = 0
        alpha = self.config.confidence_smoothing_alpha
        memory.ema_confidence = (
            alpha * track.confidence + (1.0 - alpha) * memory.ema_confidence
        )
        memory.directions.append(track.direction)
        memory.last_stable_direction = _majority_direction(
            memory.directions,
            memory.last_stable_direction,
        )
        if memory.consecutive_hits >= self.config.confirmation_frames:
            memory.was_confirmed = True
        return memory


def smooth_confidence(values: list[float], alpha: float) -> float:
    """Exponential moving average used by tests and documentation."""
    if not values:
        raise ValueError("Cannot smooth an empty confidence sequence.")
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
    smoothed = values[0]
    for value in values[1:]:
        smoothed = alpha * value + (1.0 - alpha) * smoothed
    return smoothed


def stabilize_direction(
    directions: list[Direction],
    previous: Direction | None = None,
) -> Direction:
    """Majority-vote helper for a short direction history."""
    if not directions:
        raise ValueError("Cannot stabilize an empty direction sequence.")
    return _majority_direction(directions, previous)


def _majority_direction(
    directions: list[Direction] | deque[Direction],
    previous: Direction | None,
) -> Direction:
    counts = Counter(directions)
    ranked = counts.most_common()
    if not ranked:
        if previous is None:
            raise ValueError("No directions available.")
        return previous
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        if previous is not None:
            return previous
    return ranked[0][0]
