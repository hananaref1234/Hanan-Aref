"""Rolling latency and FPS statistics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceStats:
    """Smoothed timings for one moment in the processing loop.

    These numbers describe the machine that produced them. Laptop results
    must not be treated as Raspberry Pi 5 performance.
    """

    inference_ms: float
    frame_ms: float
    fps: float
    tracking_ms: float | None = None


class PerformanceMonitor:
    """Keep a rolling average of the last N inference and frame times."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}.")
        self._inference_ms: deque[float] = deque(maxlen=window_size)
        self._frame_ms: deque[float] = deque(maxlen=window_size)
        self._tracking_ms: deque[float] = deque(maxlen=window_size)

    def record(
        self,
        inference_ms: float,
        frame_ms: float,
        tracking_ms: float | None = None,
    ) -> PerformanceStats:
        """Store one sample and return the current rolling averages."""
        self._inference_ms.append(inference_ms)
        self._frame_ms.append(frame_ms)
        if tracking_ms is not None:
            self._tracking_ms.append(tracking_ms)

        avg_inference = _mean(self._inference_ms)
        avg_frame = _mean(self._frame_ms)
        fps = 1000.0 / avg_frame if avg_frame > 0 else 0.0
        avg_tracking = _mean(self._tracking_ms) if self._tracking_ms else None
        return PerformanceStats(
            inference_ms=avg_inference,
            frame_ms=avg_frame,
            fps=fps,
            tracking_ms=avg_tracking,
        )


def _mean(values: deque[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
