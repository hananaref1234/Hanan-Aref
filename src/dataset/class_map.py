"""Training-ID to global Li-Stick taxonomy mapping.

Custom experiments may train on a contiguous subset of classes (for example
stairs+curb as 0/1) without rewriting the project-wide taxonomy (IDs 0–11).
The sidecar file next to a checkpoint translates those training IDs back to
global class IDs and names for ``ObjectDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.dataset.taxonomy import (
    CLASS_TO_ID,
    COMBINED_CLASS_TO_ID,
    COMBINED_CLASSES,
    COMBINED_DETECTOR_CLASSES,
    TAXONOMY_VERSION,
)


class ClassMapError(ValueError):
    """Raised when a training class-map file is missing or invalid."""


@dataclass(frozen=True)
class MappedClass:
    training_id: int
    name: str
    global_id: int


@dataclass(frozen=True)
class TrainingClassMap:
    model_version: str
    taxonomy_version: str
    dataset_version: str
    entries: tuple[MappedClass, ...]

    def training_to_global(self) -> dict[int, int]:
        return {entry.training_id: entry.global_id for entry in self.entries}

    def global_to_training(self) -> dict[int, int]:
        return {entry.global_id: entry.training_id for entry in self.entries}

    def training_names(self) -> dict[int, str]:
        return {entry.training_id: entry.name for entry in self.entries}

    def global_names(self) -> dict[int, str]:
        return {entry.global_id: entry.name for entry in self.entries}

    def by_training_id(self) -> dict[int, MappedClass]:
        return {entry.training_id: entry for entry in self.entries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "taxonomy_version": self.taxonomy_version,
            "dataset_version": self.dataset_version,
            "trained_classes": [entry.name for entry in self.entries],
            "training_to_global": {
                str(entry.training_id): {
                    "name": entry.name,
                    "global_id": entry.global_id,
                }
                for entry in self.entries
            },
            "global_to_training": {
                str(entry.global_id): {
                    "name": entry.name,
                    "training_id": entry.training_id,
                }
                for entry in self.entries
            },
            "notes": (
                "Training IDs are contiguous for Ultralytics. "
                "Custom global IDs 0–11 match taxonomy v0.2 in "
                "data/li_stick_navigation/dataset.yaml. Generic global IDs start at 12. "
                "This file does not change the project-wide custom taxonomy."
            ),
        }


def exp1_stairs_curb_class_map() -> TrainingClassMap:
    """li-stick-navigation-v0.2-exp1: stairs and curb only."""
    return TrainingClassMap(
        model_version="li-stick-navigation-v0.2-exp1",
        taxonomy_version=TAXONOMY_VERSION,
        dataset_version="li-stick-navigation-v0.2",
        entries=(
            MappedClass(training_id=0, name="stairs", global_id=CLASS_TO_ID["stairs"]),
            MappedClass(training_id=1, name="curb", global_id=CLASS_TO_ID["curb"]),
        ),
    )


def exp2_six_class_class_map() -> TrainingClassMap:
    """li-stick-navigation-v0.2-exp2: six custom hazard classes."""
    return _six_class_map("li-stick-navigation-v0.2-exp2")


def exp2b_six_class_class_map() -> TrainingClassMap:
    """li-stick-navigation-v0.2-exp2b: same six classes, new split."""
    return _six_class_map("li-stick-navigation-v0.2-exp2b")


def v03_combined_class_map() -> TrainingClassMap:
    """li-stick-navigation-v0.3-combined: six custom hazards plus seven generics."""
    entries = []
    for training_id, name in enumerate(COMBINED_DETECTOR_CLASSES):
        entries.append(
            MappedClass(
                training_id=training_id,
                name=name,
                global_id=COMBINED_CLASS_TO_ID[name],
            )
        )
    return TrainingClassMap(
        model_version="li-stick-navigation-v0.3-combined",
        taxonomy_version=TAXONOMY_VERSION,
        dataset_version="li-stick-navigation-v0.3-combined",
        entries=tuple(entries),
    )


def _six_class_map(model_version: str) -> TrainingClassMap:
    return TrainingClassMap(
        model_version=model_version,
        taxonomy_version=TAXONOMY_VERSION,
        dataset_version="li-stick-navigation-v0.2",
        entries=(
            MappedClass(training_id=0, name="pole", global_id=CLASS_TO_ID["pole"]),
            MappedClass(training_id=1, name="bollard", global_id=CLASS_TO_ID["bollard"]),
            MappedClass(training_id=2, name="stairs", global_id=CLASS_TO_ID["stairs"]),
            MappedClass(training_id=3, name="curb", global_id=CLASS_TO_ID["curb"]),
            MappedClass(training_id=4, name="barrier", global_id=CLASS_TO_ID["barrier"]),
            MappedClass(training_id=5, name="tree_trunk", global_id=CLASS_TO_ID["tree_trunk"]),
        ),
    )


def sidecar_path_for_checkpoint(model_path: Path) -> Path:
    """Return ``<stem>.class_map.yaml`` next to a checkpoint."""
    return model_path.with_name(f"{model_path.stem}.class_map.yaml")


def write_class_map(path: Path, class_map: TrainingClassMap) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(class_map.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def load_class_map(path: Path) -> TrainingClassMap:
    """Load and validate a training class-map YAML file."""
    if not path.exists():
        raise ClassMapError(f"Class map not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ClassMapError(f"Malformed class map {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ClassMapError(f"{path} must contain a YAML mapping.")

    training_to_global = raw.get("training_to_global")
    if not isinstance(training_to_global, dict) or not training_to_global:
        raise ClassMapError(f"{path} must contain a non-empty training_to_global mapping.")

    entries: list[MappedClass] = []
    for key, value in training_to_global.items():
        if not isinstance(value, dict):
            raise ClassMapError(f"training_to_global[{key!r}] must be a mapping.")
        try:
            training_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ClassMapError(f"Invalid training id {key!r} in {path}.") from exc
        name = str(value.get("name", "")).strip()
        global_id = value.get("global_id")
        if name not in COMBINED_CLASS_TO_ID:
            raise ClassMapError(
                f"Class {name!r} is not in the v0.2 custom taxonomy or the "
                f"combined generic set. Valid names: {', '.join(COMBINED_CLASSES)}."
            )
        if not isinstance(global_id, int) or isinstance(global_id, bool):
            raise ClassMapError(f"global_id for training id {training_id} must be an int.")
        expected_global = COMBINED_CLASS_TO_ID[name]
        if global_id != expected_global:
            raise ClassMapError(
                f"Class {name!r} must map to global id {expected_global}, got {global_id}."
            )
        entries.append(MappedClass(training_id=training_id, name=name, global_id=global_id))

    entries.sort(key=lambda item: item.training_id)
    training_ids = [item.training_id for item in entries]
    if training_ids != list(range(len(entries))):
        raise ClassMapError(
            "Training IDs must be contiguous starting at 0 "
            f"(Ultralytics requirement). Found {training_ids}."
        )

    return TrainingClassMap(
        model_version=str(raw.get("model_version", "")).strip(),
        taxonomy_version=str(raw.get("taxonomy_version", "")).strip() or TAXONOMY_VERSION,
        dataset_version=str(raw.get("dataset_version", "")).strip(),
        entries=tuple(entries),
    )


def load_sidecar_class_map(model_path: Path) -> TrainingClassMap | None:
    """Load a checkpoint sidecar if present; otherwise return None."""
    sidecar = sidecar_path_for_checkpoint(model_path)
    if not sidecar.exists():
        return None
    return load_class_map(sidecar)


def apply_class_map(
    training_id: int,
    class_name: str,
    class_map: TrainingClassMap | None,
) -> tuple[int, str]:
    """Translate a training-head id/name to global Li-Stick id/name.

    If no class map is loaded, values pass through unchanged (pretrained COCO).
    """
    if class_map is None:
        return training_id, class_name
    entry = class_map.by_training_id().get(training_id)
    if entry is None:
        return training_id, class_name
    return entry.global_id, entry.name
