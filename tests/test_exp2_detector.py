"""Exp2 checkpoint integration: class remap plus detector→tracker path.

The live YOLO load is skipped when the registered .pt is absent (CI without
weights). Class-map remapping is always tested.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from src.dataset.class_map import apply_class_map, load_class_map
from src.tracking.event_suppressor import EventSuppressor
from src.tracking.stabilization import TemporalStabilizer
from src.tracking.tracker import ObjectTracker
from src.utils.config_loader import load_app_config
from src.utils.paths import project_root
from src.vision.detector import ObjectDetector
from src.vision.detection import Detection

CHECKPOINT = project_root() / "models" / "li_stick_navigation_v0.2_exp2.pt"
CLASS_MAP = project_root() / "models" / "li_stick_navigation_v0.2_exp2.class_map.yaml"


def test_exp2_class_map_sidecar_remaps_training_ids() -> None:
    class_map = load_class_map(CLASS_MAP)
    assert apply_class_map(0, "pole", class_map) == (0, "pole")
    assert apply_class_map(1, "bollard", class_map) == (1, "bollard")
    assert apply_class_map(2, "stairs", class_map) == (2, "stairs")
    assert apply_class_map(3, "curb", class_map) == (3, "curb")
    assert apply_class_map(4, "barrier", class_map) == (6, "barrier")
    assert apply_class_map(5, "tree_trunk", class_map) == (11, "tree_trunk")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Exp2 checkpoint not registered")
def test_exp2_objectdetector_pipeline_on_test_image() -> None:
    config = load_app_config(
        project_root() / "config" / "config_v02_exp2.yaml",
        project_root() / "config" / "classes_v02_exp2.yaml",
    )
    detector = ObjectDetector(
        model_path=config.detector.model_path,
        confidence_threshold=0.10,
        allowed_classes=config.allowed_classes,
        image_size=config.detector.image_size,
    )
    assert detector._class_map is not None
    assert detector._class_map.training_to_global()[4] == 6
    assert detector._class_map.training_to_global()[5] == 11

    image_dir = project_root() / "data" / "training" / "li_stick_v02_exp2" / "images" / "test"
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not images:
        pytest.skip("Exp2 test images are not on disk.")
    frame = cv2.imread(str(images[0]))
    assert frame is not None

    detections, inference_ms = detector.detect(frame, timestamp=1.0)
    assert inference_ms >= 0.0
    assert all(isinstance(item, Detection) for item in detections)
    allowed = set(config.allowed_classes or [])
    global_ids = {0, 1, 2, 3, 6, 11}
    for item in detections:
        assert item.class_name in allowed
        assert item.class_id in global_ids

    if not detections:
        from src.vision.direction import Direction

        detections = [
            Detection(
                class_id=6,
                class_name="barrier",
                confidence=0.9,
                x1=100,
                y1=80,
                x2=220,
                y2=400,
                center_x=160,
                center_y=240,
                direction=Direction.LEFT,
                timestamp=1.0,
            )
        ]

    tracker = ObjectTracker(config.tracking)
    stabilizer = TemporalStabilizer(config.tracking)
    suppressor = EventSuppressor(config.events)
    tracks = tracker.update(detections, frame=frame)
    visible, _lost = stabilizer.update(tracks)
    events = suppressor.update(visible, [])
    assert isinstance(tracks, list)
    assert isinstance(visible, list)
    assert isinstance(events, list)
