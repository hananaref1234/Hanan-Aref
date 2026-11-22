"""Training configuration for the custom Li-Stick detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import project_root, resolve_from_root


class TrainingConfigError(ValueError):
    """Raised when training.yaml is missing or invalid."""


@dataclass(frozen=True)
class AugmentationConfig:
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    fliplr: float
    flipud: float
    mosaic: float
    mixup: float
    erasing: float


@dataclass(frozen=True)
class TrainingRunConfig:
    base_model: str
    epochs: int
    image_size: int
    batch_size: int
    patience: int
    device: str
    workers: int
    dataset_yaml: Path
    project: Path
    name: str
    register: Path | None
    class_map: Path | None
    augmentation: AugmentationConfig


def load_training_config(path: str | Path | None = None) -> TrainingRunConfig:
    resolved = (
        Path(path)
        if path is not None
        else project_root() / "config" / "training.yaml"
    )
    if not resolved.is_absolute():
        resolved = resolve_from_root(resolved)
    if not resolved.exists():
        raise TrainingConfigError(
            f"Training config not found: {resolved}. "
            "Create config/training.yaml or pass --config."
        )
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise TrainingConfigError(f"Malformed training YAML {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TrainingConfigError(f"{resolved} must contain a YAML mapping.")

    model = raw.get("model") or {}
    training = raw.get("training") or {}
    dataset = raw.get("dataset") or {}
    output = raw.get("output") or {}
    augmentation = raw.get("augmentation") or {}
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise TrainingConfigError("model and training sections must be mappings.")

    base_model = str(model.get("base_model", "yolo26n.pt")).strip()
    if not base_model:
        raise TrainingConfigError("model.base_model must be a non-empty string.")

    dataset_yaml = dataset.get("yaml", "data/li_stick_navigation/dataset.yaml")
    if not dataset_yaml:
        raise TrainingConfigError("dataset.yaml path is required.")
    dataset_path = Path(str(dataset_yaml))
    if not dataset_path.is_absolute():
        dataset_path = resolve_from_root(dataset_path)

    project = Path(str(output.get("project", "runs/li_stick")))
    if not project.is_absolute():
        project = resolve_from_root(project)

    register = _optional_path(output.get("register"))
    class_map = _optional_path(output.get("class_map"))
    if class_map is None and dataset_path.parent.joinpath("class_map.yaml").exists():
        class_map = dataset_path.parent / "class_map.yaml"

    return TrainingRunConfig(
        base_model=base_model,
        epochs=_positive_int(training.get("epochs", 100), "training.epochs"),
        image_size=_positive_int(training.get("image_size", 640), "training.image_size"),
        batch_size=_positive_int(training.get("batch_size", 16), "training.batch_size"),
        patience=_non_negative_int(training.get("patience", 20), "training.patience"),
        device=str(training.get("device", "auto")),
        workers=_non_negative_int(training.get("workers", 2), "training.workers"),
        dataset_yaml=dataset_path,
        project=project,
        name=str(output.get("name", "navigation_v1")),
        register=register,
        class_map=class_map,
        augmentation=_parse_augmentation(augmentation),
    )


def resolve_device(requested: str) -> str | int:
    """Map 'auto' to cuda:0 or cpu. Never assume a GPU is present."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized in {"cuda", "gpu", "0"}:
        return 0
    if normalized.isdigit():
        return int(normalized)
    return requested


def _parse_augmentation(data: Any) -> AugmentationConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TrainingConfigError("augmentation must be a mapping.")
    flipud = float(data.get("flipud", 0.0))
    if flipud != 0.0:
        raise TrainingConfigError(
            "augmentation.flipud must be 0. Vertical flips reverse the appearance "
            "of stairs and curbs and are not allowed."
        )
    return AugmentationConfig(
        hsv_h=float(data.get("hsv_h", 0.015)),
        hsv_s=float(data.get("hsv_s", 0.4)),
        hsv_v=float(data.get("hsv_v", 0.4)),
        degrees=float(data.get("degrees", 5.0)),
        translate=float(data.get("translate", 0.08)),
        scale=float(data.get("scale", 0.25)),
        shear=float(data.get("shear", 0.0)),
        perspective=float(data.get("perspective", 0.0004)),
        fliplr=float(data.get("fliplr", 0.5)),
        flipud=0.0,
        mosaic=float(data.get("mosaic", 0.3)),
        mixup=float(data.get("mixup", 0.0)),
        erasing=float(data.get("erasing", 0.15)),
    )


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = resolve_from_root(path)
    return path


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingConfigError(f"{field} must be a positive integer.")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrainingConfigError(f"{field} must be an integer >= 0.")
    return value
