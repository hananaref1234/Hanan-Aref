"""Unit tests for YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from src.utils.config_loader import ConfigError, load_app_config
from src.utils.paths import project_root


def _write_config(directory: Path, body: str, name: str = "config.yaml") -> Path:
    path = directory / name
    path.write_text(dedent(body).lstrip(), encoding="utf-8")
    return path


def _write_classes(directory: Path, names: list[str] | None = None) -> Path:
    names = names if names is not None else ["person", "chair"]
    listed = "\n".join(f"  - {name}" for name in names)
    path = directory / "classes.yaml"
    path.write_text(f"allowed_classes:\n{listed}\n", encoding="utf-8")
    return path


def test_load_project_default_config() -> None:
    config = load_app_config()
    assert config.config_path == project_root() / "config" / "config.yaml"
    assert config.camera.backend == "opencv"
    assert config.camera.width == 1280
    assert config.camera.height == 720
    assert config.detector.model_path == "yolo26n.pt"
    assert config.detector.confidence_threshold == 0.50
    assert config.detector.image_size == 640
    assert config.detector.use_class_filter is True
    assert config.allowed_classes is not None
    assert "person" in config.allowed_classes
    assert "dining table" in config.allowed_classes
    assert "stairs_up" not in config.allowed_classes
    assert config.tracking.enabled is True
    assert config.tracking.tracker == "bytetrack"
    assert config.tracking.confirmation_frames == 3
    assert config.tracking.max_missing_frames == 5
    assert config.events.suppress_duplicates is True
    assert config.events.duplicate_cooldown_seconds == 3.0


def test_valid_custom_config(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        camera:
          backend: opencv
          device: 1
          width: 640
          height: 480
          fps: 15
        detector:
          model_path: yolo26n.pt
          confidence_threshold: 0.25
          image_size: 320
          use_class_filter: true
        visualization:
          enabled: false
          show_fps: false
          show_latency: true
          show_direction_zones: false
        logging:
          level: DEBUG
        """,
    )
    classes_path = _write_classes(tmp_path, ["person", "car"])
    config = load_app_config(config_path, classes_path)
    assert config.camera.device == 1
    assert config.camera.width == 640
    assert config.detector.confidence_threshold == 0.25
    assert config.visualization.enabled is False
    assert config.logging.level == "DEBUG"
    assert config.allowed_classes == ["person", "car"]
    assert config.tracking.enabled is True
    assert config.tracking.confirmation_frames == 3


def test_class_filter_can_be_disabled(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        camera:
          backend: opencv
          device: 0
          width: 1280
          height: 720
          fps: 30
        detector:
          model_path: yolo26n.pt
          confidence_threshold: 0.5
          image_size: 640
          use_class_filter: false
        visualization:
          enabled: true
          show_fps: true
          show_latency: true
          show_direction_zones: true
        logging:
          level: INFO
        """,
    )
    config = load_app_config(config_path, tmp_path / "missing-classes.yaml")
    assert config.allowed_classes is None
    assert config.classes_path is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("confidence_threshold", -0.1, "confidence_threshold"),
        ("confidence_threshold", 1.1, "confidence_threshold"),
        ("image_size", 0, "image_size"),
        ("image_size", -640, "image_size"),
    ],
)
def test_invalid_detector_values(
    tmp_path: Path,
    field: str,
    value: float,
    match: str,
) -> None:
    config_path = _write_config(
        tmp_path,
        f"""
        camera:
          backend: opencv
          device: 0
          width: 1280
          height: 720
          fps: 30
        detector:
          model_path: yolo26n.pt
          confidence_threshold: {0.5 if field != "confidence_threshold" else value}
          image_size: {640 if field != "image_size" else int(value)}
          use_class_filter: false
        visualization:
          enabled: true
          show_fps: true
          show_latency: true
          show_direction_zones: true
        logging:
          level: INFO
        """,
    )
    with pytest.raises(ConfigError, match=match):
        load_app_config(config_path)


@pytest.mark.parametrize(
    ("width", "height", "match"),
    [
        (0, 720, "camera.width"),
        (-1, 720, "camera.width"),
        (1280, 0, "camera.height"),
        (1280, -10, "camera.height"),
    ],
)
def test_invalid_camera_dimensions(
    tmp_path: Path,
    width: int,
    height: int,
    match: str,
) -> None:
    config_path = _write_config(
        tmp_path,
        f"""
        camera:
          backend: opencv
          device: 0
          width: {width}
          height: {height}
          fps: 30
        detector:
          model_path: yolo26n.pt
          confidence_threshold: 0.5
          image_size: 640
          use_class_filter: false
        visualization:
          enabled: true
          show_fps: true
          show_latency: true
          show_direction_zones: true
        logging:
          level: INFO
        """,
    )
    with pytest.raises(ConfigError, match=match):
        load_app_config(config_path)


def test_unknown_backend(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        camera:
          backend: mystery
          device: 0
          width: 1280
          height: 720
          fps: 30
        detector:
          model_path: yolo26n.pt
          confidence_threshold: 0.5
          image_size: 640
          use_class_filter: false
        visualization:
          enabled: true
          show_fps: true
          show_latency: true
          show_direction_zones: true
        logging:
          level: INFO
        """,
    )
    with pytest.raises(ConfigError, match="camera.backend"):
        load_app_config(config_path)


def test_missing_config_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError, match="not found"):
        load_app_config(missing)


def test_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("camera: [unterminated\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Malformed YAML"):
        load_app_config(path)


def test_empty_yaml(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_app_config(path)


_BASE_CONFIG = """
camera:
  backend: opencv
  device: 0
  width: 1280
  height: 720
  fps: 30
detector:
  model_path: yolo26n.pt
  confidence_threshold: 0.5
  image_size: 640
  use_class_filter: false
visualization:
  enabled: true
  show_fps: true
  show_latency: true
  show_direction_zones: true
logging:
  level: INFO
"""


def test_invalid_tracking_values(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _BASE_CONFIG
        + """
tracking:
  enabled: true
  tracker: bytetrack
  confirmation_frames: 0
  max_missing_frames: 5
  confidence_smoothing_alpha: 0.4
  direction_history_size: 5
""",
    )
    with pytest.raises(ConfigError, match="confirmation_frames"):
        load_app_config(config_path)


def test_invalid_smoothing_alpha(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _BASE_CONFIG
        + """
tracking:
  enabled: true
  tracker: bytetrack
  confirmation_frames: 3
  max_missing_frames: 5
  confidence_smoothing_alpha: 1.4
  direction_history_size: 5
""",
    )
    with pytest.raises(ConfigError, match="confidence_smoothing_alpha"):
        load_app_config(config_path)


def test_invalid_cooldown(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _BASE_CONFIG
        + """
events:
  suppress_duplicates: true
  duplicate_cooldown_seconds: -1
  emit_direction_changes: true
  log_lost_tracks: true
""",
    )
    with pytest.raises(ConfigError, match="duplicate_cooldown_seconds"):
        load_app_config(config_path)


def test_unknown_tracker(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _BASE_CONFIG
        + """
tracking:
  enabled: true
  tracker: mystery
  confirmation_frames: 3
  max_missing_frames: 5
  confidence_smoothing_alpha: 0.4
  direction_history_size: 5
""",
    )
    with pytest.raises(ConfigError, match="tracking.tracker"):
        load_app_config(config_path)
