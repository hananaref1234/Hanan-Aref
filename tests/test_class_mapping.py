"""Class mapping safety: semantic stairs/curb, no invented direction."""

from __future__ import annotations

import pytest

from src.dataset.importers.mapping import (
    MappingError,
    assert_auto_mapping_allowed,
    decide_mapping,
    normalize_class_name,
)
from src.dataset.manifest import get_source


def test_sideguide_direct_and_review_mappings() -> None:
    source = get_source("sideguide")
    pole = decide_mapping(source, "pole")
    assert pole.mapped_class == "pole"
    assert pole.mapping_type == "DIRECT"
    assert pole.review_status == "PENDING"

    trunk = decide_mapping(source, "tree trunk")
    assert trunk.mapped_class == "tree_trunk"
    assert trunk.mapping_type == "DIRECT"

    barricade = decide_mapping(source, "barricade")
    assert barricade.proposed_class == "barrier"
    assert barricade.mapped_class is None
    assert barricade.mapping_type == "SEMANTIC_REVIEW"

    person = decide_mapping(source, "person")
    assert person.proposed_class == "person"
    assert person.taxonomy == "combined"
    assert person.mapped_class is None


def test_stairs_and_curbs_are_semantic_not_directional() -> None:
    source = get_source("accessibility_barriers")
    stairs = decide_mapping(source, "stairs")
    assert stairs.mapped_class == "stairs"
    assert stairs.mapping_type == "DIRECT"
    assert stairs.review_status == "APPROVED"
    assert stairs.mapped_class not in {"stairs_up", "stairs_down"}

    stair = decide_mapping(source, "stair")
    assert stair.mapped_class == "stairs"
    assert stair.review_status == "APPROVED"

    curb_step = decide_mapping(source, "step", extra_attributes={"type": "curb"})
    assert curb_step.mapped_class == "curb"
    assert curb_step.review_status == "APPROVED"
    assert curb_step.mapped_class not in {"curb_up", "curb_down"}

    doorstep = decide_mapping(source, "step", extra_attributes={"type": "doorstep"})
    assert doorstep.review_status == "UNMAPPED_SOURCE_CLASS"
    assert doorstep.mapped_class is None

    other_step = decide_mapping(source, "step", extra_attributes={"type": "other"})
    assert other_step.review_status == "UNMAPPED_SOURCE_CLASS"

    generic_step = decide_mapping(source, "step")
    assert generic_step.review_status == "UNMAPPED_SOURCE_CLASS"

    ramp = decide_mapping(source, "ramp")
    assert ramp.review_status == "UNMAPPED_SOURCE_CLASS"
    assert ramp.mapped_class is None

    grab = decide_mapping(source, "grab bar")
    assert grab.review_status == "UNMAPPED_SOURCE_CLASS"


def test_open_door_is_not_auto_open_doorway() -> None:
    source = get_source("doordetect")
    opened = decide_mapping(source, "open")
    assert opened.proposed_class == "open_doorway"
    assert opened.mapped_class is None
    assert opened.mapping_type == "SEMANTIC_REVIEW"

    semi = decide_mapping(source, "semi-open")
    assert semi.proposed_class is None
    assert semi.mapped_class is None


def test_vegetation_cannot_map_to_tree_trunk() -> None:
    source = get_source("cityscapes")
    vegetation = decide_mapping(source, "vegetation")
    assert vegetation.mapped_class is None
    assert vegetation.proposed_class is None
    with pytest.raises(MappingError, match="Refusing automatic mapping"):
        assert_auto_mapping_allowed("vegetation", "tree_trunk")


def test_forbidden_directional_auto_pairs() -> None:
    for original, target in (
        ("stairs", "stairs_up"),
        ("stairs", "stairs_down"),
        ("curb", "curb_up"),
        ("step", "curb_down"),
        ("open", "open_doorway"),
        ("Tree", "tree_trunk"),
        ("street light", "pole"),
        ("traffic sign", "pole"),
        ("fence", "barrier"),
    ):
        with pytest.raises(MappingError):
            assert_auto_mapping_allowed(original, target)


def test_normalize_tree_trunk_aliases() -> None:
    assert normalize_class_name("Tree-Trunk") == "tree trunk"
    assert normalize_class_name("tree_trunk") == "tree trunk"
