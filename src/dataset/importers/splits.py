"""Deterministic train/val/test assignment by unique source image."""

from __future__ import annotations

import random
from dataclasses import replace

from src.dataset.importers.provenance import ProvenanceRecord

DEFAULT_SPLIT_SEED = 42
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
VALID_SPLITS = ("train", "val", "test")


def assign_splits_by_source_image(
    records: list[ProvenanceRecord],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> list[ProvenanceRecord]:
    """Assign every box from the same source_image to one split.

    Existing train/val/test source_split or detector_split values are kept.
    Remaining images are sorted, then shuffled with a fixed seed.
    """
    existing: dict[str, str] = {}
    unsplit: set[str] = set()
    for record in records:
        split = _normalized_split(record.detector_split or record.source_split)
        if record.source_image and split in VALID_SPLITS:
            previous = existing.get(record.source_image)
            if previous and previous != split:
                raise ValueError(
                    f"Source image {record.source_image!r} already has split "
                    f"{previous!r} and {split!r}."
                )
            existing[record.source_image] = split
        elif record.source_image:
            unsplit.add(record.source_image)
    unsplit -= set(existing)
    split_for_image = dict(existing)
    if unsplit:
        shuffled = sorted(unsplit)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        for index, image in enumerate(shuffled):
            split_for_image[image] = _split_name(index, len(shuffled), ratios)
    updated: list[ProvenanceRecord] = []
    for record in records:
        split = split_for_image.get(record.source_image, "")
        if not split:
            updated.append(record)
            continue
        updated.append(replace(record, detector_split=split))
    _assert_no_image_split_leakage(updated)
    return updated


def split_counts(records: list[ProvenanceRecord]) -> dict[str, dict[str, int]]:
    """Count unique images and boxes per assigned detector split."""
    images: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    boxes = {"train": 0, "val": 0, "test": 0}
    for record in records:
        split = _normalized_split(record.detector_split or record.source_split)
        if split not in images:
            continue
        images[split].add(record.source_image)
        boxes[split] += 1
    return {
        split: {"images": len(images[split]), "annotations": boxes[split]}
        for split in VALID_SPLITS
    }


def _normalized_split(value: str) -> str:
    split = (value or "").lower()
    if split in {"valid", "validation"}:
        return "val"
    return split


def _split_name(index: int, total: int, ratios: tuple[float, float, float]) -> str:
    train_n, val_n, _test_n = _split_sizes(total, ratios)
    if index < train_n:
        return "train"
    if index < train_n + val_n:
        return "val"
    return "test"


def _split_sizes(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    train_ratio, val_ratio, test_ratio = ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}.")
    n_val = int(round(total * val_ratio))
    n_test = int(round(total * test_ratio))
    if total >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    n_train = total - n_val - n_test
    while n_train + n_val + n_test > total and n_train > 0:
        n_train -= 1
    while n_train + n_val + n_test < total:
        n_train += 1
    return n_train, n_val, n_test


def _assert_no_image_split_leakage(records: list[ProvenanceRecord]) -> None:
    by_image: dict[str, set[str]] = {}
    for record in records:
        split = _normalized_split(record.detector_split)
        if not record.source_image or not split:
            continue
        by_image.setdefault(record.source_image, set()).add(split)
    leaks = {image: splits for image, splits in by_image.items() if len(splits) > 1}
    if leaks:
        sample = next(iter(leaks.items()))
        raise ValueError(
            f"Source image {sample[0]!r} assigned to multiple splits: "
            f"{sorted(sample[1])}."
        )
