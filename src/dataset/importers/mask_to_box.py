"""Derive YOLO-style boxes from semantic/instance masks.

This conversion is lossy. Callers must record mapping_type MASK_TO_BOX and
must not treat the result as equivalent to the original mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dataset.importers.boxes import PixelBox

MASK_TO_BOX_LOSS = (
    "Mask-to-box conversion discards pixel-precise shape. Concave regions "
    "become axis-aligned rectangles that include background. Adjacent "
    "instances of the same semantic class without instance IDs are merged. "
    "Holes and thin gaps disappear. A wall spanning the frame becomes one "
    "huge box that is a weak detection target."
)


@dataclass(frozen=True)
class MaskBox:
    class_id: int
    box: PixelBox
    pixel_count: int
    component_index: int


def boxes_from_label_map(
    label_map: np.ndarray,
    class_ids: set[int],
    *,
    min_pixels: int = 32,
) -> list[MaskBox]:
    """Connected-component boxes for selected class IDs in a 2D label image.

    ``label_map`` is a 2D integer array (H, W). Instance IDs are not required.
    Same-class pixels that touch (8-connected) become one box.
    """
    if label_map.ndim != 2:
        raise ValueError("label_map must be a 2D array.")
    boxes: list[MaskBox] = []
    for class_id in sorted(class_ids):
        binary = (label_map == class_id).astype(np.uint8)
        for index, (box, pixels) in enumerate(_connected_boxes(binary, min_pixels)):
            boxes.append(
                MaskBox(
                    class_id=class_id,
                    box=box,
                    pixel_count=pixels,
                    component_index=index,
                )
            )
    return boxes


def boxes_from_instance_map(
    instance_map: np.ndarray,
    *,
    min_pixels: int = 32,
    ignore_ids: set[int] | None = None,
) -> list[MaskBox]:
    """One box per instance ID (non-zero) in a 2D instance image."""
    if instance_map.ndim != 2:
        raise ValueError("instance_map must be a 2D array.")
    skip = ignore_ids or {0}
    boxes: list[MaskBox] = []
    for instance_id in np.unique(instance_map):
        value = int(instance_id)
        if value in skip:
            continue
        binary = (instance_map == value).astype(np.uint8)
        components = _connected_boxes(binary, min_pixels)
        for index, (box, pixels) in enumerate(components):
            boxes.append(
                MaskBox(
                    class_id=value,
                    box=box,
                    pixel_count=pixels,
                    component_index=index,
                )
            )
    return boxes


def _connected_boxes(
    binary: np.ndarray,
    min_pixels: int,
) -> list[tuple[PixelBox, int]]:
    import cv2

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    results: list[tuple[PixelBox, int]] = []
    # Label 0 is background.
    for index in range(1, num_labels):
        x, y, width, height, area = stats[index]
        if int(area) < min_pixels:
            continue
        results.append(
            (
                PixelBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + width),
                    y2=float(y + height),
                ),
                int(area),
            )
        )
    return results
