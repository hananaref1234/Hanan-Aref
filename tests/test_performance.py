"""Unit tests for rolling performance statistics (no hardware)."""

from __future__ import annotations

import pytest

from src.monitoring.performance import PerformanceMonitor


def test_rolling_average_and_fps() -> None:
    monitor = PerformanceMonitor(window_size=3)
    first = monitor.record(inference_ms=10.0, frame_ms=20.0)
    assert first.inference_ms == 10.0
    assert first.frame_ms == 20.0
    assert first.fps == pytest.approx(50.0)

    second = monitor.record(inference_ms=30.0, frame_ms=20.0)
    assert second.inference_ms == 20.0
    assert second.fps == pytest.approx(50.0)


def test_invalid_window() -> None:
    with pytest.raises(ValueError, match="window_size"):
        PerformanceMonitor(window_size=0)
