"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import project_root, resolve_from_root

_VALID_CAMERA_BACKENDS = frozenset({"opencv", "picamera2"})
_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})
_VALID_TRACKERS = frozenset({"bytetrack"})


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class CameraConfig:
    backend: str
    device: int | str
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str
    confidence_threshold: float
    image_size: int
    use_class_filter: bool


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool
    show_fps: bool
    show_latency: bool
    show_direction_zones: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    tracker: str
    confirmation_frames: int
    max_missing_frames: int
    confidence_smoothing_alpha: float
    direction_history_size: int


@dataclass(frozen=True)
class EventsConfig:
    suppress_duplicates: bool
    duplicate_cooldown_seconds: float
    emit_direction_changes: bool
    log_lost_tracks: bool


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    detector: DetectorConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
    tracking: TrackingConfig
    events: EventsConfig
    allowed_classes: list[str] | None
    config_path: Path
    classes_path: Path | None


def load_app_config(
    config_path: str | Path | None = None,
    classes_path: str | Path | None = None,
) -> AppConfig:
    """Load and validate ``config.yaml`` plus optional ``classes.yaml``.

    Parameters
    ----------
    config_path:
        Path to the main YAML file. Defaults to ``config/config.yaml``.
    classes_path:
        Path to the class-filter YAML file. Defaults to ``config/classes.yaml``.
    """
    resolved_config = (
        Path(config_path)
        if config_path is not None
        else project_root() / "config" / "config.yaml"
    )
    if not resolved_config.is_absolute():
        resolved_config = resolve_from_root(resolved_config)

    raw = _read_yaml(resolved_config, kind="configuration")

    camera = _parse_camera(raw.get("camera"))
    detector = _parse_detector(raw.get("detector"))
    visualization = _parse_visualization(raw.get("visualization"))
    logging_cfg = _parse_logging(raw.get("logging"))
    tracking = _parse_tracking(raw.get("tracking"))
    events = _parse_events(raw.get("events"))

    resolved_classes: Path | None = None
    allowed_classes: list[str] | None = None
    if detector.use_class_filter:
        resolved_classes = (
            Path(classes_path)
            if classes_path is not None
            else project_root() / "config" / "classes.yaml"
        )
        if not resolved_classes.is_absolute():
            resolved_classes = resolve_from_root(resolved_classes)
        allowed_classes = _parse_allowed_classes(resolved_classes)

    return AppConfig(
        camera=camera,
        detector=detector,
        visualization=visualization,
        logging=logging_cfg,
        tracking=tracking,
        events=events,
        allowed_classes=allowed_classes,
        config_path=resolved_config,
        classes_path=resolved_classes,
    )


def _read_yaml(path: Path, kind: str) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"{kind.capitalize()} file not found: {path}. "
            "Create the file or pass a valid --config path."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Malformed YAML in {kind} file {path}: {exc}. "
            "Check indentation and quoting."
        ) from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {kind} file {path}: {exc}.") from exc

    if data is None:
        raise ConfigError(f"{kind.capitalize()} file {path} is empty.")
    if not isinstance(data, dict):
        raise ConfigError(
            f"{kind.capitalize()} file {path} must contain a YAML mapping at the top level."
        )
    return data


def _require_mapping(section: Any, name: str) -> dict[str, Any]:
    if section is None:
        raise ConfigError(f"Missing required '{name}' section in configuration.")
    if not isinstance(section, dict):
        raise ConfigError(f"Configuration section '{name}' must be a mapping.")
    return section


def _parse_camera(section: Any) -> CameraConfig:
    data = _require_mapping(section, "camera")
    backend = str(data.get("backend", "opencv")).strip().lower()
    if backend not in _VALID_CAMERA_BACKENDS:
        raise ConfigError(
            f"Invalid camera.backend {backend!r}. "
            f"Use one of: {', '.join(sorted(_VALID_CAMERA_BACKENDS))}."
        )

    device = data.get("device", 0)
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    if not isinstance(device, (int, str)):
        raise ConfigError(
            "camera.device must be a camera index (int) or a video file path (str)."
        )

    width = _positive_int(data.get("width"), "camera.width")
    height = _positive_int(data.get("height"), "camera.height")
    fps = _positive_number(data.get("fps"), "camera.fps")

    return CameraConfig(
        backend=backend,
        device=device,
        width=width,
        height=height,
        fps=float(fps),
    )


def _parse_detector(section: Any) -> DetectorConfig:
    data = _require_mapping(section, "detector")
    model_path = data.get("model_path")
    if not model_path or not isinstance(model_path, str):
        raise ConfigError(
            "detector.model_path must be a non-empty string such as 'yolo26n.pt'."
        )

    confidence = data.get("confidence_threshold")
    if not isinstance(confidence, (int, float)):
        raise ConfigError("detector.confidence_threshold must be a number between 0 and 1.")
    confidence_f = float(confidence)
    if confidence_f < 0 or confidence_f > 1:
        raise ConfigError(
            f"detector.confidence_threshold must be between 0 and 1 inclusive, got {confidence_f}."
        )

    image_size = _positive_int(data.get("image_size"), "detector.image_size")
    use_class_filter = bool(data.get("use_class_filter", True))

    return DetectorConfig(
        model_path=model_path,
        confidence_threshold=confidence_f,
        image_size=image_size,
        use_class_filter=use_class_filter,
    )


def _parse_visualization(section: Any) -> VisualizationConfig:
    data = _require_mapping(section, "visualization")
    return VisualizationConfig(
        enabled=bool(data.get("enabled", True)),
        show_fps=bool(data.get("show_fps", True)),
        show_latency=bool(data.get("show_latency", True)),
        show_direction_zones=bool(data.get("show_direction_zones", True)),
    )


def _parse_tracking(section: Any) -> TrackingConfig:
    if section is None:
        data: dict[str, Any] = {}
    elif not isinstance(section, dict):
        raise ConfigError("Configuration section 'tracking' must be a mapping.")
    else:
        data = section

    tracker = str(data.get("tracker", "bytetrack")).strip().lower()
    if tracker not in _VALID_TRACKERS:
        raise ConfigError(
            f"Invalid tracking.tracker {tracker!r}. "
            f"Use one of: {', '.join(sorted(_VALID_TRACKERS))}."
        )

    alpha = data.get("confidence_smoothing_alpha", 0.4)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ConfigError("tracking.confidence_smoothing_alpha must be a number in (0, 1].")
    alpha_f = float(alpha)
    if alpha_f <= 0 or alpha_f > 1:
        raise ConfigError(
            f"tracking.confidence_smoothing_alpha must be in (0, 1], got {alpha_f}."
        )

    return TrackingConfig(
        enabled=bool(data.get("enabled", True)),
        tracker=tracker,
        confirmation_frames=_positive_int(
            data.get("confirmation_frames", 3), "tracking.confirmation_frames"
        ),
        max_missing_frames=_positive_int(
            data.get("max_missing_frames", 5), "tracking.max_missing_frames"
        ),
        confidence_smoothing_alpha=alpha_f,
        direction_history_size=_positive_int(
            data.get("direction_history_size", 5), "tracking.direction_history_size"
        ),
    )


def _parse_events(section: Any) -> EventsConfig:
    if section is None:
        data: dict[str, Any] = {}
    elif not isinstance(section, dict):
        raise ConfigError("Configuration section 'events' must be a mapping.")
    else:
        data = section

    cooldown = data.get("duplicate_cooldown_seconds", 3.0)
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)):
        raise ConfigError("events.duplicate_cooldown_seconds must be a number >= 0.")
    cooldown_f = float(cooldown)
    if cooldown_f < 0:
        raise ConfigError(
            f"events.duplicate_cooldown_seconds must be >= 0, got {cooldown_f}."
        )

    return EventsConfig(
        suppress_duplicates=bool(data.get("suppress_duplicates", True)),
        duplicate_cooldown_seconds=cooldown_f,
        emit_direction_changes=bool(data.get("emit_direction_changes", True)),
        log_lost_tracks=bool(data.get("log_lost_tracks", True)),
    )


def _parse_logging(section: Any) -> LoggingConfig:
    data = _require_mapping(section, "logging")
    level = str(data.get("level", "INFO")).upper()
    if level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"Invalid logging.level {level!r}. "
            f"Use one of: {', '.join(sorted(_VALID_LOG_LEVELS))}."
        )
    return LoggingConfig(level=level)


def _parse_allowed_classes(path: Path) -> list[str]:
    raw = _read_yaml(path, kind="classes")
    classes = raw.get("allowed_classes")
    if classes is None:
        raise ConfigError(
            f"classes file {path} must define an 'allowed_classes' list, "
            "or set detector.use_class_filter to false."
        )
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ConfigError(f"'allowed_classes' in {path} must be a list of strings.")
    cleaned = [item.strip() for item in classes if item.strip()]
    if not cleaned:
        raise ConfigError(
            f"'allowed_classes' in {path} is empty. "
            "Add class names or disable detector.use_class_filter."
        )
    return cleaned


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field} must be a positive integer.")
    if value <= 0:
        raise ConfigError(f"{field} must be > 0, got {value}.")
    return value


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a positive number.")
    if value <= 0:
        raise ConfigError(f"{field} must be > 0, got {value}.")
    return float(value)
