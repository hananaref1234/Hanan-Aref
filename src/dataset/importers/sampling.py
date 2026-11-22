"""Deterministic subsampling for large public sources.

Prefer whole-image selection to preserve annotation consistency and avoid
splitting instances from the same frame across different sample sets.
"""

from __future__ import annotations

import hashlib
from typing import TypeVar

T = TypeVar("T")


def deterministic_rank(key: str, *, seed: int) -> str:
    """Stable sort key derived from seed + logical key."""
    payload = f"{seed}:{key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_items_by_rank(items: list[T], *, key_fn, max_items: int, seed: int) -> list[T]:
    """Return up to ``max_items`` items in deterministic pseudo-random order."""
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    ranked = sorted(items, key=lambda item: deterministic_rank(key_fn(item), seed=seed))
    return ranked[:max_items]


def cap_instances_by_image(
    grouped: dict[str, list[T]],
    *,
    max_instances: int,
    seed: int,
) -> list[T]:
    """Keep whole image groups until ``max_instances`` is reached.

    ``grouped`` maps an image key to the objects parsed from that image.
    """
    if max_instances <= 0:
        flattened: list[T] = []
        for key in sorted(grouped):
            flattened.extend(grouped[key])
        return flattened

    image_keys = list(grouped)
    selected_keys = select_items_by_rank(
        image_keys,
        key_fn=lambda key: key,
        max_items=len(image_keys),
        seed=seed,
    )
    kept: list[T] = []
    total = 0
    for key in selected_keys:
        batch = grouped[key]
        if total >= max_instances:
            break
        if total + len(batch) > max_instances and total > 0:
            continue
        kept.extend(batch)
        total += len(batch)
    return kept
