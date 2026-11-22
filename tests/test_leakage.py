"""Split-leakage helper tests."""

from __future__ import annotations

from src.dataset.leakage import MetadataRow, find_split_leakage, infer_sequence_key


def test_same_video_in_two_splits_is_leakage() -> None:
    rows = [
        MetadataRow(filename="a.jpg", split="train", video_id="walk01", sequence_id="walk01"),
        MetadataRow(filename="b.jpg", split="test", video_id="walk01", sequence_id="walk01"),
    ]
    leaks = find_split_leakage(rows)
    assert leaks
    fields = {item.group_field for item in leaks}
    assert "video_id" in fields
    assert "sequence_id" in fields


def test_different_videos_are_not_leakage() -> None:
    rows = [
        MetadataRow(filename="a.jpg", split="train", video_id="walk01"),
        MetadataRow(filename="b.jpg", split="test", video_id="walk02"),
    ]
    assert find_split_leakage(rows) == []


def test_same_source_image_in_two_splits_is_leakage() -> None:
    rows = [
        MetadataRow(filename="a.jpg", split="train", sequence_id="images/0001.jpg"),
        MetadataRow(filename="b.jpg", split="val", sequence_id="images/0001.jpg"),
    ]
    leaks = find_split_leakage(rows)
    assert any(item.group_field == "sequence_id" for item in leaks)


def test_infer_sequence_key_from_frame_filename() -> None:
    assert infer_sequence_key("campus_north_frame_000123.jpg") == "campus_north_frame"
    assert infer_sequence_key("unique_still.jpg") == "unique_still"
