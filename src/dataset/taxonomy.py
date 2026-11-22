"""Li-Stick custom-class taxonomy.

Taxonomy v0.1 used directional detector classes (stairs_up/down, curb_up/down).
Taxonomy v0.2 is the active detector taxonomy: those four classes are replaced
by semantic `stairs` and `curb`. Direction is preserved as metadata only.

Do not silently rewrite v0.1. Do not reorder v0.2 after labeling begins.
"""

from __future__ import annotations

from pathlib import Path

import yaml

TAXONOMY_VERSION_V0_1 = "v0.1"
TAXONOMY_VERSION = "v0.2"

# Historical detector classes. Kept for review metadata and tests.
CUSTOM_CLASSES_V0_1: tuple[str, ...] = (
    "pole",
    "bollard",
    "stairs_up",
    "stairs_down",
    "curb_up",
    "curb_down",
    "open_doorway",
    "closed_door",
    "barrier",
    "drop_off",
    "pothole",
    "low_overhead_obstacle",
    "wall",
    "tree_trunk",
)

DIRECTIONAL_CLASSES_V0_1: tuple[str, ...] = (
    "stairs_up",
    "stairs_down",
    "curb_up",
    "curb_down",
)

# Active detector classes (taxonomy v0.2).
CUSTOM_CLASSES: tuple[str, ...] = (
    "pole",
    "bollard",
    "stairs",
    "curb",
    "open_doorway",
    "closed_door",
    "barrier",
    "drop_off",
    "pothole",
    "low_overhead_obstacle",
    "wall",
    "tree_trunk",
)

CLASS_TO_ID: dict[str, int] = {name: index for index, name in enumerate(CUSTOM_CLASSES)}
ID_TO_CLASS: dict[int, str] = {index: name for index, name in enumerate(CUSTOM_CLASSES)}

# Combined-head generics (v0.3-combined). Custom IDs 0–11 stay frozen.
# couch / dining table are not in this experiment; bus and truck are.
GENERIC_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
)

# Combined single-model head: frozen custom IDs first, generics appended.
# Fine-tuning only the custom classes would replace the COCO head, so
# pretrained generic outputs are not assumed to survive custom-only training.
GENERIC_START_ID = len(CUSTOM_CLASSES)
GENERIC_CLASS_TO_ID: dict[str, int] = {
    name: GENERIC_START_ID + index for index, name in enumerate(GENERIC_CLASSES)
}
GENERIC_ID_TO_CLASS: dict[int, str] = {
    class_id: name for name, class_id in GENERIC_CLASS_TO_ID.items()
}
COMBINED_CLASSES: tuple[str, ...] = CUSTOM_CLASSES + GENERIC_CLASSES
COMBINED_CLASS_TO_ID: dict[str, int] = {**CLASS_TO_ID, **GENERIC_CLASS_TO_ID}
COMBINED_ID_TO_CLASS: dict[int, str] = {**ID_TO_CLASS, **GENERIC_ID_TO_CLASS}

# v0.3 detector training subset (contiguous YOLO IDs, not the sparse global map).
COMBINED_DETECTOR_CLASSES: tuple[str, ...] = (
    "pole",
    "bollard",
    "stairs",
    "curb",
    "barrier",
    "tree_trunk",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
)

# yolo26n.pt is COCO-pretrained (80 classes). Inspected checkpoint names: no "tree"
# and no "tree_trunk". Closest COCO class is "potted plant", which is not a
# navigation tree trunk. tree_trunk is therefore a custom class (ID 11), not a
# pretrained generic class.
PRETRAINED_HAS_TREE = False
CUSTOM_CLASS_CANDIDATES: tuple[str, ...] = ()

SAFETY_CRITICAL_CLASSES: tuple[str, ...] = (
    "drop_off",
    "low_overhead_obstacle",
)

# Kept as bounding-box classes, but bbox-only detection is expected to be weak.
BBOX_DIFFICULT_CLASSES: tuple[str, ...] = (
    "stairs",
    "curb",
    "drop_off",
    "open_doorway",
)

DIRECTION_TO_DETECTOR_CLASS: dict[str, str] = {
    "stairs_up": "stairs",
    "stairs_down": "stairs",
    "curb_up": "curb",
    "curb_down": "curb",
}

DATASET_VERSION = "li-stick-navigation-v0.2"


def detector_class_from_direction(name: str) -> str | None:
    """Map a v0.1 directional class onto the v0.2 detector class."""
    return DIRECTION_TO_DETECTOR_CLASS.get(name)


def custom_class_id(name: str) -> int:
    """Return the frozen ID for a custom class name (taxonomy v0.2)."""
    if name not in CLASS_TO_ID:
        raise ValueError(
            f"Unknown custom class {name!r}. "
            f"Valid names: {', '.join(CUSTOM_CLASSES)}."
        )
    return CLASS_TO_ID[name]


def custom_class_name(class_id: int) -> str:
    """Return the frozen name for a custom class ID (taxonomy v0.2)."""
    if class_id not in ID_TO_CLASS:
        raise ValueError(
            f"Unknown custom class id {class_id}. "
            f"Valid ids: 0-{len(CUSTOM_CLASSES) - 1}."
        )
    return ID_TO_CLASS[class_id]


def combined_class_id(name: str) -> int:
    """Return the global ID for a custom or combined-generic class name."""
    if name not in COMBINED_CLASS_TO_ID:
        raise ValueError(
            f"Unknown combined class {name!r}. "
            f"Valid names: {', '.join(COMBINED_CLASSES)}."
        )
    return COMBINED_CLASS_TO_ID[name]


def combined_class_name(class_id: int) -> str:
    """Return the name for a global custom or combined-generic class ID."""
    if class_id not in COMBINED_ID_TO_CLASS:
        raise ValueError(
            f"Unknown combined class id {class_id}. "
            f"Valid ids: 0-{len(COMBINED_CLASSES) - 1}."
        )
    return COMBINED_ID_TO_CLASS[class_id]


def load_dataset_yaml(path: Path) -> dict:
    """Load a YOLO dataset YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def load_dataset_yaml_names(path: Path) -> dict[int, str]:
    """Load class id → name mapping from a YOLO dataset YAML file."""
    data = load_dataset_yaml(path)
    if "names" not in data:
        raise ValueError(f"{path} must contain a 'names' mapping or list.")
    names = data["names"]
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    raise ValueError(f"'names' in {path} must be a list or mapping.")


def assert_dataset_yaml_matches_taxonomy(path: Path) -> None:
    """Fail if dataset.yaml class order drifted from CUSTOM_CLASSES v0.2."""
    data = load_dataset_yaml(path)
    names = load_dataset_yaml_names(path)
    expected = dict(enumerate(CUSTOM_CLASSES))
    if names != expected:
        raise ValueError(
            f"dataset.yaml class mapping does not match taxonomy {TAXONOMY_VERSION}.\n"
            f"Expected: {expected}\n"
            f"Found:    {names}"
        )
    version = str(data.get("taxonomy_version", "")).strip()
    if version != TAXONOMY_VERSION:
        raise ValueError(
            f"dataset.yaml taxonomy_version must be {TAXONOMY_VERSION!r}, "
            f"found {version!r}."
        )
