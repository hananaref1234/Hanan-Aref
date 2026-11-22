"""Deterministic sampling helpers."""

from __future__ import annotations

from src.dataset.importers.sampling import cap_instances_by_image, select_items_by_rank


def test_select_items_by_rank_is_deterministic() -> None:
    items = [f"img_{index}" for index in range(20)]
    first = select_items_by_rank(items, key_fn=lambda item: item, max_items=5, seed=42)
    second = select_items_by_rank(items, key_fn=lambda item: item, max_items=5, seed=42)
    assert first == second
    assert len(first) == 5


def test_cap_instances_by_image_preserves_whole_images() -> None:
    grouped = {
        "a.jpg": ["obj1", "obj2"],
        "b.jpg": ["obj3"],
        "c.jpg": ["obj4", "obj5", "obj6"],
    }
    kept = cap_instances_by_image(grouped, max_instances=3, seed=7)
    assert kept
    assert all(isinstance(item, str) for item in kept)
    assert len(kept) <= 6
