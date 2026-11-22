"""Unit tests for the Detection dataclass (no YOLO inference)."""

from __future__ import annotations

from src.vision.detection import Detection
from src.vision.direction import Direction


def _sample_detection() -> Detection:
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.916,
        x1=210,
        y1=92,
        x2=421,
        y2=610,
        center_x=315,
        center_y=351,
        direction=Direction.CENTER,
        timestamp=1760000000.123,
    )


def test_detection_fields() -> None:
    detection = _sample_detection()
    assert detection.class_name == "person"
    assert detection.confidence == 0.916
    assert detection.direction is Direction.CENTER
    assert detection.bbox.x1 == 210
    assert detection.bbox.y2 == 610


def test_detection_to_dict_structure() -> None:
    payload = _sample_detection().to_dict()
    assert payload == {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.916,
        "bbox": {"x1": 210, "y1": 92, "x2": 421, "y2": 610},
        "center": {"x": 315, "y": 351},
        "direction": "CENTER",
        "timestamp": 1760000000.123,
    }


def test_detection_log_line() -> None:
    line = _sample_detection().log_line()
    assert line == (
        "[DETECTION] class=person confidence=0.92 direction=CENTER bbox=(210,92,421,610)"
    )


def test_future_fields_are_absent() -> None:
    payload = _sample_detection().to_dict()
    for key in ("track_id", "risk_level", "distance", "source", "relevance_score", "depth_m"):
        assert key not in payload


def test_depth_m_is_optional_and_not_invented() -> None:
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        x1=1,
        y1=1,
        x2=2,
        y2=2,
        center_x=1,
        center_y=1,
        direction=Direction.CENTER,
        timestamp=1.0,
        depth_m=None,
    )
    assert detection.depth_m is None
    assert "depth_m" not in detection.to_dict()
