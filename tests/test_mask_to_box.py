"""Mask-to-box conversion on tiny synthetic arrays."""

from __future__ import annotations

import numpy as np

from src.dataset.importers.mask_to_box import (
    MASK_TO_BOX_LOSS,
    boxes_from_instance_map,
    boxes_from_label_map,
)


def test_separate_components_make_two_boxes() -> None:
    label_map = np.zeros((32, 32), dtype=np.uint8)
    label_map[2:10, 2:6] = 17  # pole
    label_map[2:10, 20:24] = 17
    boxes = boxes_from_label_map(label_map, {17}, min_pixels=4)
    assert len(boxes) == 2
    assert all(box.class_id == 17 for box in boxes)


def test_adjacent_same_class_pixels_merge() -> None:
    label_map = np.zeros((16, 16), dtype=np.uint8)
    label_map[4:8, 4:8] = 12
    label_map[4:8, 8:12] = 12
    boxes = boxes_from_label_map(label_map, {12}, min_pixels=4)
    assert len(boxes) == 1
    assert boxes[0].box.width == 8


def test_instance_map_keeps_separate_ids() -> None:
    instance_map = np.zeros((16, 16), dtype=np.int32)
    instance_map[1:4, 1:4] = 101
    instance_map[1:4, 8:11] = 102
    boxes = boxes_from_instance_map(instance_map, min_pixels=4)
    assert {box.class_id for box in boxes} == {101, 102}


def test_tiny_noise_is_dropped() -> None:
    label_map = np.zeros((16, 16), dtype=np.uint8)
    label_map[0, 0] = 17
    assert boxes_from_label_map(label_map, {17}, min_pixels=8) == []


def test_loss_documentation_mentions_merged_instances() -> None:
    assert "merged" in MASK_TO_BOX_LOSS.lower() or "Adjacent" in MASK_TO_BOX_LOSS
