"""Frozen custom-class taxonomy and dataset.yaml alignment."""

from __future__ import annotations

import pytest
import yaml

from src.dataset.taxonomy import (
    COMBINED_CLASSES,
    CUSTOM_CLASS_CANDIDATES,
    CUSTOM_CLASSES,
    CUSTOM_CLASSES_V0_1,
    DIRECTIONAL_CLASSES_V0_1,
    GENERIC_CLASSES,
    GENERIC_START_ID,
    PRETRAINED_HAS_TREE,
    TAXONOMY_VERSION,
    TAXONOMY_VERSION_V0_1,
    assert_dataset_yaml_matches_taxonomy,
    custom_class_id,
    custom_class_name,
    combined_class_id,
    detector_class_from_direction,
    load_dataset_yaml_names,
)
from src.utils.paths import project_root

EXPECTED_V0_1_IDS = {
    0: "pole",
    1: "bollard",
    2: "stairs_up",
    3: "stairs_down",
    4: "curb_up",
    5: "curb_down",
    6: "open_doorway",
    7: "closed_door",
    8: "barrier",
    9: "drop_off",
    10: "pothole",
    11: "low_overhead_obstacle",
    12: "wall",
    13: "tree_trunk",
}

EXPECTED_CUSTOM_IDS = {
    0: "pole",
    1: "bollard",
    2: "stairs",
    3: "curb",
    4: "open_doorway",
    5: "closed_door",
    6: "barrier",
    7: "drop_off",
    8: "pothole",
    9: "low_overhead_obstacle",
    10: "wall",
    11: "tree_trunk",
}


def test_v0_1_directional_taxonomy_is_preserved() -> None:
    assert TAXONOMY_VERSION_V0_1 == "v0.1"
    assert dict(enumerate(CUSTOM_CLASSES_V0_1)) == EXPECTED_V0_1_IDS
    assert CUSTOM_CLASSES_V0_1[2:6] == DIRECTIONAL_CLASSES_V0_1
    assert "stairs_up" not in CUSTOM_CLASSES
    assert "stairs" not in CUSTOM_CLASSES_V0_1


def test_frozen_custom_class_order_v0_2() -> None:
    assert TAXONOMY_VERSION == "v0.2"
    assert dict(enumerate(CUSTOM_CLASSES)) == EXPECTED_CUSTOM_IDS
    assert CUSTOM_CLASSES[10] == "wall"
    assert CUSTOM_CLASSES[11] == "tree_trunk"
    assert len(CUSTOM_CLASSES) == 12
    assert custom_class_id("stairs") == 2
    assert custom_class_id("curb") == 3
    assert custom_class_id("wall") == 10
    assert custom_class_id("tree_trunk") == 11
    assert custom_class_name(4) == "open_doorway"
    assert custom_class_name(11) == "tree_trunk"


def test_v0_1_direction_maps_to_semantic_detector_classes() -> None:
    assert detector_class_from_direction("stairs_up") == "stairs"
    assert detector_class_from_direction("stairs_down") == "stairs"
    assert detector_class_from_direction("curb_up") == "curb"
    assert detector_class_from_direction("curb_down") == "curb"


def test_tree_trunk_is_custom_not_pretrained() -> None:
    assert PRETRAINED_HAS_TREE is False
    assert "tree" not in CUSTOM_CLASSES
    assert "tree" not in GENERIC_CLASSES
    assert "tree_trunk" in CUSTOM_CLASSES
    assert "tree" not in CUSTOM_CLASS_CANDIDATES


def test_combined_taxonomy_starts_after_frozen_custom_ids() -> None:
    assert GENERIC_START_ID == 12
    assert COMBINED_CLASSES[:12] == CUSTOM_CLASSES
    assert COMBINED_CLASSES[12] == "person"
    assert COMBINED_CLASSES[16] == "bus"
    assert COMBINED_CLASSES[17] == "truck"
    assert COMBINED_CLASSES[18] == "chair"
    assert combined_class_id("person") == 12
    assert combined_class_id("bus") == 16
    assert combined_class_id("truck") == 17
    assert combined_class_id("chair") == 18
    assert "custom_person" not in COMBINED_CLASSES
    assert "dining_table" not in COMBINED_CLASSES
    assert "couch" not in COMBINED_CLASSES
    assert "dining table" not in COMBINED_CLASSES


def test_generic_classes_not_duplicated_as_custom() -> None:
    overlap = set(CUSTOM_CLASSES) & set(GENERIC_CLASSES)
    assert overlap == set()


def test_unknown_class_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown custom class"):
        custom_class_id("stairs_up")
    with pytest.raises(ValueError, match="Unknown custom class id"):
        custom_class_name(99)


def test_repo_dataset_yaml_matches_taxonomy() -> None:
    path = project_root() / "data" / "li_stick_navigation" / "dataset.yaml"
    assert_dataset_yaml_matches_taxonomy(path)
    names = load_dataset_yaml_names(path)
    assert names == EXPECTED_CUSTOM_IDS


def test_all_class_yaml_files_share_the_same_ids() -> None:
    root = project_root()
    dataset_names = load_dataset_yaml_names(root / "data" / "li_stick_navigation" / "dataset.yaml")
    custom_list = yaml.safe_load(
        (root / "config" / "classes_custom.yaml").read_text(encoding="utf-8")
    )["allowed_classes"]
    combined_list = yaml.safe_load(
        (root / "config" / "classes_combined.yaml").read_text(encoding="utf-8")
    )["allowed_classes"]

    assert dataset_names == EXPECTED_CUSTOM_IDS
    assert custom_list == list(CUSTOM_CLASSES)
    assert combined_list[: len(CUSTOM_CLASSES)] == list(CUSTOM_CLASSES)
    assert combined_list[len(CUSTOM_CLASSES) :] == list(GENERIC_CLASSES)
