"""Tests for Raspberry Pi 5 deployment readiness.

Covers:
  - Pi config file parses correctly
  - Headless mode (visualization.enabled = false)
  - Relative path resolution works cross-platform
  - Picamera2 backend import guard (no crash on non-Pi)
  - Pi selfcheck script is importable
  - Pi benchmark script is importable
  - No Roboflow import in runtime path
  - Class map loads and contains all 13 v0.3-combined classes
  - Model file and sidecar exist
  - run_rpi5_live.py is importable
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pi config
# ---------------------------------------------------------------------------

def test_pi_config_exists() -> None:
    assert (ROOT / "config" / "config_rpi5_combined.yaml").exists()


def test_pi_config_parses(tmp_path: Path) -> None:
    from src.utils.config_loader import load_app_config

    classes = tmp_path / "classes.yaml"
    classes.write_text("allowed_classes:\n  - person\n  - barrier\n", encoding="utf-8")
    cfg = load_app_config(
        ROOT / "config" / "config_rpi5_combined.yaml",
        classes,
    )
    assert cfg.camera.backend == "picamera2"
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 720
    assert cfg.camera.fps == 30.0
    assert cfg.detector.model_path == "models/li_stick_navigation_v0.3_combined.pt"
    assert cfg.detector.image_size == 640
    assert cfg.detector.confidence_threshold == pytest.approx(0.40)
    assert cfg.visualization.enabled is False


def test_pi_config_headless_by_default() -> None:
    from src.utils.config_loader import load_app_config

    cfg = load_app_config(
        ROOT / "config" / "config_rpi5_combined.yaml",
        ROOT / "config" / "classes_v03_combined.yaml",
    )
    assert cfg.visualization.enabled is False, (
        "config_rpi5_combined.yaml must default to visualization.enabled: false "
        "for headless Pi deployment."
    )


# ---------------------------------------------------------------------------
# Headless mode: cv2 window calls must be guarded
# ---------------------------------------------------------------------------

def test_main_no_gui_calls_when_headless(tmp_path: Path) -> None:
    """When renderer is None (viz disabled), main.py must not call cv2.imshow."""
    import ast

    main_src = (ROOT / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(main_src)

    imshow_calls: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "imshow"
            ):
                imshow_calls.append(node.lineno)

    # All imshow calls must be inside an `if renderer` block — validated by
    # checking the source text directly for the guarded pattern.
    src_lines = main_src.splitlines()
    for lineno in imshow_calls:
        # Look backwards for a guard
        block_lines = src_lines[max(0, lineno - 10) : lineno]
        guarded = any(
            "renderer" in line and ("if" in line or "is not None" in line)
            for line in block_lines
        )
        assert guarded, (
            f"cv2.imshow on line {lineno} of main.py is not guarded by "
            "'if renderer is not None'. Headless Pi run would crash."
        )


# ---------------------------------------------------------------------------
# Relative path resolution
# ---------------------------------------------------------------------------

def test_resolve_from_root_relative() -> None:
    from src.utils.paths import resolve_from_root

    resolved = resolve_from_root(Path("models/li_stick_navigation_v0.3_combined.pt"))
    assert resolved.is_absolute()
    assert "li_stick_ai" in str(resolved) or resolved.parents[0].name == "models"


def test_resolve_from_root_absolute(tmp_path: Path) -> None:
    from src.utils.paths import resolve_from_root

    # Absolute paths pass through unchanged
    f = tmp_path / "foo.pt"
    f.touch()
    assert resolve_from_root(f) == f


def test_project_root_is_repo_root() -> None:
    from src.utils.paths import project_root

    root = project_root()
    assert (root / "src").is_dir()
    assert (root / "main.py").exists()


# ---------------------------------------------------------------------------
# Picamera2 guard
# ---------------------------------------------------------------------------

def test_picamera2_camera_importable_on_non_pi() -> None:
    from src.camera.picamera2_camera import Picamera2, Picamera2Camera, Picamera2Error

    cam = Picamera2Camera(width=640, height=480, fps=15)
    if Picamera2 is None:
        with pytest.raises(Picamera2Error, match="Picamera2 is not installed"):
            cam.open()
    else:
        cam.close()  # already closed, should not raise


def test_picamera2_to_bgr_rgb_conversion() -> None:
    """_to_bgr must flip RGB to BGR correctly."""
    import numpy as np

    from src.camera.picamera2_camera import Picamera2Camera

    cam = Picamera2Camera(width=640, height=480, fps=15)
    cam._rgb_input = True

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 100  # R channel
    bgr = cam._to_bgr(rgb)
    # After RGB→BGR the original R channel is now in position 2
    assert bgr[:, :, 2].mean() == pytest.approx(100)
    assert bgr[:, :, 0].mean() == pytest.approx(0)


def test_picamera2_to_bgr_rgba_conversion() -> None:
    import numpy as np

    from src.camera.picamera2_camera import Picamera2Camera

    cam = Picamera2Camera(width=640, height=480, fps=15)
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    bgr = cam._to_bgr(rgba)
    assert bgr.shape == (4, 4, 3)


def test_picamera2_to_bgr_rejects_bad_channels() -> None:
    import numpy as np

    from src.camera.picamera2_camera import Picamera2Camera, Picamera2Error

    cam = Picamera2Camera(width=640, height=480, fps=15)
    bad = np.zeros((4, 4, 2), dtype=np.uint8)
    with pytest.raises(Picamera2Error):
        cam._to_bgr(bad)


# ---------------------------------------------------------------------------
# Scripts importable
# ---------------------------------------------------------------------------

def _add_root_to_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def test_pi_selfcheck_importable() -> None:
    _add_root_to_path()
    spec = importlib.util.spec_from_file_location(  # type: ignore[attr-defined]
        "pi_selfcheck", ROOT / "scripts" / "pi_selfcheck.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    # Just loading (not executing main) must not raise
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "main")


def test_pi_benchmark_importable() -> None:
    _add_root_to_path()
    spec = importlib.util.spec_from_file_location(
        "pi_benchmark", ROOT / "scripts" / "pi_benchmark.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "main")


def test_run_rpi5_live_importable() -> None:
    _add_root_to_path()
    spec = importlib.util.spec_from_file_location(
        "run_rpi5_live", ROOT / "scripts" / "run_rpi5_live.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "main")


def test_run_detector_example_importable() -> None:
    _add_root_to_path()
    spec = importlib.util.spec_from_file_location(
        "run_detector", ROOT / "examples" / "run_detector.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "main")


# ---------------------------------------------------------------------------
# No Roboflow in runtime path
# ---------------------------------------------------------------------------

def test_no_roboflow_in_runtime_imports() -> None:
    """Core runtime modules must not import roboflow."""
    runtime_modules = [
        "src.vision.detector",
        "src.vision.detection",
        "src.camera.factory",
        "src.camera.picamera2_camera",
        "src.tracking.tracker",
        "src.tracking.stabilization",
        "src.tracking.event_suppressor",
        "src.utils.config_loader",
        "src.utils.paths",
        "src.monitoring.performance",
    ]
    for mod_name in runtime_modules:
        mod = importlib.import_module(mod_name)
        src_file = getattr(mod, "__file__", None)
        if src_file is None:
            continue
        source = Path(src_file).read_text(encoding="utf-8")
        assert "roboflow" not in source.lower(), (
            f"{mod_name} imports roboflow — runtime code must not depend on "
            "Roboflow SDK. Training-only imports belong in src/dataset/ only."
        )


# ---------------------------------------------------------------------------
# Class map and model files
# ---------------------------------------------------------------------------

def test_combined_model_exists() -> None:
    assert (ROOT / "models" / "li_stick_navigation_v0.3_combined.pt").exists(), (
        "Trained checkpoint not found. Run training before deployment."
    )


def test_combined_class_map_exists() -> None:
    assert (ROOT / "models" / "li_stick_navigation_v0.3_combined.class_map.yaml").exists()


def test_combined_class_map_has_13_classes() -> None:
    from src.dataset.class_map import load_class_map

    cm = load_class_map(
        ROOT / "models" / "li_stick_navigation_v0.3_combined.class_map.yaml"
    )
    assert len(cm.entries) == 13
    names = {e.name for e in cm.entries}
    expected = {
        "pole", "bollard", "stairs", "curb", "barrier", "tree_trunk",
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "chair",
    }
    assert names == expected


def test_combined_class_map_training_ids_contiguous() -> None:
    from src.dataset.class_map import load_class_map

    cm = load_class_map(
        ROOT / "models" / "li_stick_navigation_v0.3_combined.class_map.yaml"
    )
    training_ids = sorted(e.training_id for e in cm.entries)
    assert training_ids == list(range(13)), (
        "Training IDs must be contiguous 0–12 for Ultralytics."
    )


def test_classes_v03_combined_has_all_13() -> None:
    import yaml

    path = ROOT / "config" / "classes_v03_combined.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = data["allowed_classes"]
    expected = [
        "pole", "bollard", "stairs", "curb", "barrier", "tree_trunk",
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "chair",
    ]
    assert sorted(allowed) == sorted(expected)
