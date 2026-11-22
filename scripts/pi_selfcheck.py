"""Raspberry Pi 5 deployment self-check.

Verifies the runtime environment before starting the live loop:
  - Python version
  - Platform
  - Model file present
  - Sidecar class map present and loads correctly
  - Config parses without error
  - Picamera2 importable
  - Ultralytics importable and model loads
  - Configured resolution and image size

Does NOT require an actual camera attached for most checks.
Model loading does require the .pt file on disk.

Usage:
    python scripts/pi_selfcheck.py
    python scripts/pi_selfcheck.py --config config/config_rpi5_combined.yaml \
                                    --classes config/classes_v03_combined.yaml
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"


def _row(status: str, label: str, detail: str = "") -> None:
    pad = 45
    marker = f"[{status}]"
    print(f"{marker:<8} {label:<{pad}} {detail}")


def check_python_version() -> bool:
    v = sys.version_info
    detail = f"{sys.version.split()[0]}"
    ok = v >= (3, 9)
    _row(_PASS if ok else _FAIL, "Python version >= 3.9", detail)
    return ok


def check_platform() -> None:
    _row(_PASS, "Platform", platform.platform())


def check_model(model_path: Path) -> bool:
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        _row(_PASS, "Model file exists", f"{model_path}  ({size_mb:.1f} MB)")
        return True
    _row(_FAIL, "Model file missing", str(model_path))
    return False


def check_class_map(model_path: Path) -> bool:
    sidecar = model_path.with_suffix(".class_map.yaml")
    if sidecar.exists():
        try:
            from src.dataset.class_map import load_class_map

            cm = load_class_map(sidecar)
            _row(_PASS, "Class map loads", f"{len(cm.entries)} classes — {sidecar.name}")
            return True
        except Exception as exc:
            _row(_FAIL, "Class map load error", str(exc))
            return False
    _row(_FAIL, "Class map sidecar missing", str(sidecar))
    return False


def check_config(config_path: Path, classes_path: Path) -> bool:
    try:
        from src.utils.config_loader import load_app_config

        cfg = load_app_config(config_path, classes_path)
        _row(_PASS, "Config parses", f"backend={cfg.camera.backend}  viz={cfg.visualization.enabled}")
        _row(
            _PASS,
            "Resolution",
            f"{cfg.camera.width}x{cfg.camera.height} @ {cfg.camera.fps:.0f} FPS",
        )
        _row(_PASS, "Image size (inference)", f"imgsz={cfg.detector.image_size}")
        classes_n = len(cfg.allowed_classes) if cfg.allowed_classes else "all"
        _row(_PASS, "Class filter", f"{classes_n} classes")
        return True
    except Exception as exc:
        _row(_FAIL, "Config parse error", str(exc))
        return False


def check_picamera2() -> bool:
    try:
        from picamera2 import Picamera2  # type: ignore[import]  # noqa: F401

        _row(_PASS, "Picamera2 importable")
        return True
    except ImportError:
        _row(_FAIL, "Picamera2 NOT importable", "sudo apt install python3-picamera2")
        return False


def check_ultralytics(model_path: Path) -> bool:
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        _row(_FAIL, "ultralytics NOT importable", "pip install ultralytics")
        return False

    if not model_path.exists():
        _row(_SKIP, "Model load (file missing)")
        return False

    try:
        t0 = time.perf_counter()
        from ultralytics import YOLO

        YOLO(str(model_path))
        ms = (time.perf_counter() - t0) * 1000
        _row(_PASS, "Model loads (Ultralytics)", f"{ms:.0f} ms")
        return True
    except Exception as exc:
        _row(_FAIL, "Model load error", str(exc))
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/config_rpi5_combined.yaml"),
    )
    p.add_argument(
        "--classes",
        type=Path,
        default=Path("config/classes_v03_combined.yaml"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from src.utils.paths import resolve_from_root

    config_path = resolve_from_root(args.config)
    classes_path = resolve_from_root(args.classes)
    model_path = resolve_from_root(Path("models/li_stick_navigation_v0.3_combined.pt"))

    print("=" * 60)
    print("Li-Stick Pi Deployment Self-Check")
    print("=" * 60)

    results: list[bool] = []
    results.append(check_python_version())
    check_platform()
    results.append(check_model(model_path))
    results.append(check_class_map(model_path))
    results.append(check_config(config_path, classes_path))
    results.append(check_picamera2())
    results.append(check_ultralytics(model_path))

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    overall = _PASS if all(results) else _FAIL
    print(f"[{overall}] {passed}/{total} checks passed")
    print("=" * 60)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
