"""Raspberry Pi 5 benchmark script.

Measures real on-device performance WITHOUT an attached camera.
Uses synthetic frames so the script can be run over SSH before any
physical setup.

Reports:
  - Model load time
  - Per-frame inference latency (mean, min, max, p95)
  - Tracker update latency (if tracking enabled)
  - Total frame latency
  - Achieved FPS
  - RAM usage (RSS)
  - CPU usage
  - Number of frames tested

Run AFTER pi_selfcheck.py confirms the environment is ready.

Usage (on the Pi):
    python scripts/pi_benchmark.py

    # Custom frame count:
    python scripts/pi_benchmark.py --frames 200

    # Match live config:
    python scripts/pi_benchmark.py \
        --config config/config_rpi5_combined.yaml \
        --classes config/classes_v03_combined.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def _try_resource() -> "tuple[object, bool]":
    try:
        import resource  # Unix only

        return resource, True
    except ImportError:
        return None, False


def _rss_mb() -> float | None:
    resource, ok = _try_resource()
    if not ok:
        return None
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # type: ignore[attr-defined]
    except Exception:
        return None


def _cpu_percent() -> float | None:
    try:
        import psutil  # optional

        return psutil.Process().cpu_percent(interval=0.1)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/config_rpi5_combined.yaml"),
        help="Main config YAML.",
    )
    p.add_argument(
        "--classes",
        type=Path,
        default=Path("config/classes_v03_combined.yaml"),
        help="Class filter YAML.",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of synthetic frames to benchmark (default: 100).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup frames excluded from statistics (default: 5).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width override (default: from config).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height override (default: from config).",
    )
    return p.parse_args()


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_d) - 1)
    return sorted_d[lo] + (k - lo) * (sorted_d[hi] - sorted_d[lo])


def main() -> int:
    from src.utils.config_loader import load_app_config
    from src.utils.logging_setup import setup_logging
    from src.utils.paths import resolve_from_root
    from src.vision.detector import ObjectDetector

    args = parse_args()
    setup_logging("WARNING")

    config = load_app_config(resolve_from_root(args.config), resolve_from_root(args.classes))

    width = args.width or config.camera.width
    height = args.height or config.camera.height
    image_size = config.detector.image_size

    print("=" * 60)
    print("Li-Stick Pi Benchmark")
    print("=" * 60)
    print(f"Frame size    : {width}x{height}")
    print(f"Image size    : {image_size}")
    print(f"Model         : {config.detector.model_path}")
    print(f"Confidence    : {config.detector.confidence_threshold}")
    print(f"Tracking      : {config.tracking.enabled}")
    print(f"Frames        : {args.frames}  (warmup: {args.warmup})")
    print()

    # --- Model load ---
    print("Loading model...", end=" ", flush=True)
    t0 = time.perf_counter()
    detector = ObjectDetector(
        model_path=config.detector.model_path,
        confidence_threshold=config.detector.confidence_threshold,
        allowed_classes=config.allowed_classes,
        image_size=image_size,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"done  ({load_ms:.0f} ms)")

    # --- Optional tracker ---
    tracker = None
    stabilizer = None
    if config.tracking.enabled:
        from src.tracking.stabilization import TemporalStabilizer
        from src.tracking.tracker import create_tracker

        tracker = create_tracker(config.tracking)
        stabilizer = TemporalStabilizer(config.tracking)

    # --- Synthetic frames ---
    rng = np.random.default_rng(42)
    inference_samples: list[float] = []
    tracking_samples: list[float] = []
    frame_samples: list[float] = []

    total = args.frames + args.warmup
    print(f"Running {total} frames ({args.warmup} warmup)...")
    for i in range(total):
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        ts = time.time()
        frame_t0 = time.perf_counter()

        detections, inf_ms = detector.detect(frame, timestamp=ts)

        track_ms: float | None = None
        if tracker is not None and stabilizer is not None:
            t_start = time.perf_counter()
            tracked = tracker.update(detections, frame)
            stabilizer.update(tracked)
            track_ms = (time.perf_counter() - t_start) * 1000

        frame_ms = (time.perf_counter() - frame_t0) * 1000

        if i >= args.warmup:
            inference_samples.append(inf_ms)
            frame_samples.append(frame_ms)
            if track_ms is not None:
                tracking_samples.append(track_ms)

    rss = _rss_mb()
    cpu = _cpu_percent()

    # --- Report ---
    def _stats(label: str, data: list[float]) -> None:
        if not data:
            return
        print(
            f"  {label:<22} mean={statistics.mean(data):>7.1f} ms  "
            f"min={min(data):>7.1f} ms  max={max(data):>7.1f} ms  "
            f"p95={_percentile(data, 95):>7.1f} ms"
        )

    fps = 1000.0 / statistics.mean(frame_samples) if frame_samples else 0.0
    print()
    print("=" * 60)
    print(f"RESULTS  ({args.frames} frames after warmup)")
    print("=" * 60)
    print(f"  {'Model load':<22} {load_ms:.0f} ms")
    _stats("Inference latency", inference_samples)
    if tracking_samples:
        _stats("Tracking latency", tracking_samples)
    _stats("Total frame latency", frame_samples)
    print(f"  {'Achieved FPS':<22} {fps:.1f}")
    if rss is not None:
        print(f"  {'RAM (RSS)':<22} {rss:.1f} MB")
    if cpu is not None:
        print(f"  {'CPU usage':<22} {cpu:.1f} %")
    print("=" * 60)
    print()
    print("NOTE: These numbers were measured on synthetic random frames.")
    print("      Real camera frames may differ. Run on the actual Pi 5 for")
    print("      representative latency and FPS numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
