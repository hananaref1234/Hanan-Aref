"""YOLO label parsing for bbox and polygon annotations."""

from __future__ import annotations

import pytest

from src.dataset.yolo_labels import (
    LabelError,
    parse_yolo_annotation_line,
    parse_yolo_line,
    parse_yolo_polygon_line,
)


def test_parse_bbox_line_still_works() -> None:
    box = parse_yolo_line("3 0.5 0.4 0.2 0.3", 1, max_class_id=10)
    assert box.class_id == 3
    assert box.x_center == pytest.approx(0.5)
    assert box.y_center == pytest.approx(0.4)
    assert box.width == pytest.approx(0.2)
    assert box.height == pytest.approx(0.3)


def test_parse_triangle_polygon() -> None:
    ann = parse_yolo_annotation_line("2 0.1 0.1 0.4 0.1 0.1 0.5", 1, max_class_id=10)
    assert ann.source_format == "POLYGON"
    assert ann.class_id == 2
    assert len(ann.polygon_points) == 3
    assert ann.x_center == pytest.approx(0.25)
    assert ann.y_center == pytest.approx(0.3)
    assert ann.width == pytest.approx(0.3)
    assert ann.height == pytest.approx(0.4)


def test_parse_many_point_polygon() -> None:
    ann = parse_yolo_annotation_line(
        "1 0.1 0.1 0.5 0.1 0.7 0.3 0.7 0.8 0.2 0.9 0.1 0.4",
        2,
        max_class_id=10,
    )
    assert ann.source_format == "POLYGON"
    assert len(ann.polygon_points) == 6
    assert ann.width == pytest.approx(0.6)
    assert ann.height == pytest.approx(0.8)


def test_polygon_conversion_math() -> None:
    ann = parse_yolo_annotation_line("4 0.2 0.3 0.8 0.4 0.6 0.9", 1, max_class_id=10)
    assert ann.x_center == pytest.approx(0.5)
    assert ann.y_center == pytest.approx(0.6)
    assert ann.width == pytest.approx(0.6)
    assert ann.height == pytest.approx(0.6)


def test_polygon_odd_coordinate_count_rejected() -> None:
    with pytest.raises(LabelError, match="even number of coordinates"):
        parse_yolo_annotation_line("1 0.1 0.1 0.2 0.2 0.3", 1, max_class_id=10)


def test_polygon_fewer_than_three_points_rejected() -> None:
    with pytest.raises(LabelError, match="at least 3 points"):
        parse_yolo_polygon_line("1 0.1 0.1 0.2 0.2", 1, max_class_id=10)


def test_polygon_outside_valid_range_rejected() -> None:
    with pytest.raises(LabelError, match="outside the allowed normalized range"):
        parse_yolo_annotation_line("1 0.1 0.1 0.4 0.1 1.5 0.2", 1, max_class_id=10)


def test_zero_area_polygon_rejected() -> None:
    with pytest.raises(LabelError, match="zero area"):
        parse_yolo_annotation_line("1 0.5 0.5 0.5 0.5 0.5 0.5", 1, max_class_id=10)


def test_class_id_validation() -> None:
    with pytest.raises(LabelError, match="Valid ids are 0-3"):
        parse_yolo_annotation_line("4 0.5 0.5 0.2 0.2", 1, max_class_id=3)


def test_bbox_parser_rejects_polygon_for_backward_compatibility() -> None:
    with pytest.raises(LabelError, match="expected 5 fields"):
        parse_yolo_line("1 0.1 0.1 0.4 0.1 0.1 0.5", 1, max_class_id=10)


def test_small_tolerance_clamps_augmented_polygon() -> None:
    ann = parse_yolo_annotation_line(
        "1 0.999 0.99 1.001 0.25 0.124 0.255 0.131 1.0018",
        1,
        max_class_id=20,
    )
    assert ann.source_format == "POLYGON"
    assert 0.0 <= ann.x_center <= 1.0
    assert 0.0 <= ann.y_center <= 1.0
