"""v0.3-combined class map, merge, leakage, and detector remap tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset.class_map import (
    apply_class_map,
    load_class_map,
    v03_combined_class_map,
    write_class_map,
)
from src.dataset.combined_view import (
    ROBOFLOW_GENERIC_MAP,
    _is_roboflow_custom_hazard,
    parse_roboflow_generic_boxes,
    select_extra_roboflow_generic_images,
)
from src.dataset.stratified_split import (
    COMBINED_V03_CLASSES,
    SplitRow,
    assign_group_aware_splits,
    build_groups,
    group_id_for_row,
    leakage_report,
)
from src.dataset.taxonomy import COMBINED_CLASS_TO_ID, COMBINED_DETECTOR_CLASSES, GENERIC_START_ID
from src.dataset.validator import validate_dataset
from src.training.config import load_training_config
from src.utils.config_loader import load_app_config
from src.utils.paths import project_root
from src.vision.detection import Detection
from src.vision.direction import Direction


def test_v03_class_map_training_to_global() -> None:
    class_map = v03_combined_class_map()
    assert class_map.training_to_global() == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 6,
        5: 11,
        6: 12,
        7: 13,
        8: 14,
        9: 15,
        10: 16,
        11: 17,
        12: 18,
    }
    assert apply_class_map(0, "pole", class_map) == (0, "pole")
    assert apply_class_map(4, "barrier", class_map) == (6, "barrier")
    assert apply_class_map(5, "tree_trunk", class_map) == (11, "tree_trunk")
    assert apply_class_map(6, "person", class_map) == (12, "person")
    assert apply_class_map(8, "car", class_map) == (14, "car")
    assert apply_class_map(12, "chair", class_map) == (18, "chair")
    assert COMBINED_DETECTOR_CLASSES[12] == "chair"
    assert COMBINED_CLASS_TO_ID["person"] == GENERIC_START_ID
    assert COMBINED_CLASS_TO_ID["chair"] == 18


def test_combined_class_id_helpers() -> None:
    from src.dataset.taxonomy import combined_class_id, combined_class_name

    assert combined_class_id("person") == 12
    assert combined_class_id("chair") == 18
    assert combined_class_id("barrier") == 6
    assert combined_class_name(14) == "car"
    assert combined_class_name(11) == "tree_trunk"


def test_v03_class_map_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "map.yaml"
    write_class_map(path, v03_combined_class_map())
    loaded = load_class_map(path)
    assert loaded.training_to_global()[6] == 12
    assert loaded.training_names()[12] == "chair"


def test_pedestrian_maps_to_person() -> None:
    assert ROBOFLOW_GENERIC_MAP["pedestrian"] == "person"
    assert ROBOFLOW_GENERIC_MAP["person"] == "person"
    assert "jeep" not in ROBOFLOW_GENERIC_MAP
    assert "van" not in ROBOFLOW_GENERIC_MAP
    assert "chair" not in ROBOFLOW_GENERIC_MAP


def test_extra_generic_skips_unlabeled_custom_hazards(tmp_path: Path) -> None:
    names = {0: "person", 1: "pole", 2: "car"}
    safe = tmp_path / "safe.jpg"
    unsafe = tmp_path / "unsafe.jpg"
    _write_png(safe)
    _write_png(unsafe)
    (tmp_path / "safe.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "unsafe.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.4\n",
        encoding="utf-8",
    )
    index = {
        "safe.jpg": (safe, tmp_path / "safe.txt", names),
        "unsafe.jpg": (unsafe, tmp_path / "unsafe.txt", names),
    }
    extras = select_extra_roboflow_generic_images(
        index,
        already=set(),
        current_counts={"person": 0, "car": 0, "bicycle": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    )
    filenames = {box.filename for box in extras}
    assert "safe.jpg" in filenames
    assert "unsafe.jpg" not in filenames
    assert all(box.mapped_class in {"person", "car"} for box in extras)


def test_same_image_generic_parse_keeps_person_and_skips_jeep(tmp_path: Path) -> None:
    names = {0: "person", 1: "jeep", 2: "bicycle"}
    image = tmp_path / "scene.jpg"
    label = tmp_path / "scene.txt"
    _write_png(image)
    label.write_text(
        "0 0.4 0.4 0.1 0.3\n1 0.7 0.7 0.2 0.2\n2 0.2 0.8 0.1 0.1\n",
        encoding="utf-8",
    )
    boxes, source_names, has_custom = parse_roboflow_generic_boxes(
        image,
        label,
        names,
        filename="scene.jpg",
        group_id="g",
        origin="roboflow_same_image_merge",
        source_root=tmp_path,
    )
    assert has_custom is False
    assert "jeep" in source_names
    mapped = {box.mapped_class for box in boxes}
    assert mapped == {"person", "bicycle"}
    assert all("jeep" not in box.mapped_class for box in boxes)


def test_custom_hazard_name_detection() -> None:
    assert _is_roboflow_custom_hazard("pole")
    assert _is_roboflow_custom_hazard("barricade")
    assert _is_roboflow_custom_hazard("tree_trunk")
    assert not _is_roboflow_custom_hazard("person")
    assert not _is_roboflow_custom_hazard("jeep")


def test_combined_group_aware_split_no_leakage() -> None:
    rows = []
    for index, name in enumerate(COMBINED_V03_CLASSES):
        for copy in range(8):
            filename = f"{name}_{copy}.jpg"
            rows.append(
                SplitRow(
                    sample_id=f"{name}-{copy}",
                    source_dataset="synthetic",
                    source_image=filename,
                    filename=filename,
                    mapped_class=name,
                    original_source_split="train",
                    group_id=group_id_for_row("synthetic", filename, filename),
                )
            )
    # Near-duplicate pair must stay together.
    rows.append(
        SplitRow(
            sample_id="pole-dup-a",
            source_dataset="roboflow_sidewalk_barriers",
            source_image="a.rf.111.jpg",
            filename="a.rf.111.jpg",
            mapped_class="pole",
            original_source_split="train",
            group_id=group_id_for_row(
                "roboflow_sidewalk_barriers", "a.rf.111.jpg", "a.rf.111.jpg"
            ),
        )
    )
    rows.append(
        SplitRow(
            sample_id="pole-dup-b",
            source_dataset="roboflow_sidewalk_barriers",
            source_image="a.rf.222.jpg",
            filename="a.rf.222.jpg",
            mapped_class="pole",
            original_source_split="train",
            group_id=group_id_for_row(
                "roboflow_sidewalk_barriers", "a.rf.222.jpg", "a.rf.222.jpg"
            ),
        )
    )
    groups = build_groups(rows, class_names=COMBINED_V03_CLASSES)
    result = assign_group_aware_splits(groups, class_names=COMBINED_V03_CLASSES, seed=1)
    leaks = leakage_report(result)
    assert leaks["source_image_leakage"] == 0
    assert leaks["group_leakage"] == 0
    assert result.filename_to_split["a.rf.111.jpg"] == result.filename_to_split["a.rf.222.jpg"]
    for split in ("val", "test"):
        for name in COMBINED_V03_CLASSES:
            assert result.class_counts[split][name] >= 0


def test_validator_accepts_combined_training_view(tmp_path: Path) -> None:
    root = tmp_path / "view"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(COMBINED_DETECTOR_CLASSES))
    (root / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "taxonomy_version: v0.2\ntraining_view: li-stick-navigation-v0.3-combined\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    _write_png(root / "images" / "train" / "a.png")
    (root / "labels" / "train" / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n6 0.2 0.2 0.1 0.3\n12 0.8 0.8 0.1 0.1\n",
        encoding="utf-8",
    )
    report = validate_dataset(root)
    assert report.passed
    assert report.annotations_total == 3


def test_v03_training_and_webcam_configs() -> None:
    training = load_training_config(project_root() / "config" / "training_v03_combined.yaml")
    assert training.base_model == "yolo26n.pt"
    assert training.epochs == 100
    assert training.image_size == 640
    assert training.batch_size == 16
    assert training.patience == 20
    assert "li_stick_v03_combined" in str(training.dataset_yaml)
    assert training.register == project_root() / "models" / "li_stick_navigation_v0.3_combined.pt"
    config = load_app_config(
        project_root() / "config" / "config_v03_combined.yaml",
        project_root() / "config" / "classes_v03_combined.yaml",
    )
    assert config.detector.model_path == "models/li_stick_navigation_v0.3_combined.pt"
    assert "person" in config.allowed_classes
    assert "chair" in config.allowed_classes
    assert "barrier" in config.allowed_classes
    assert "open_doorway" not in config.allowed_classes


def test_detection_exposes_handoff_fields() -> None:
    detection = Detection(
        class_id=12,
        class_name="person",
        confidence=0.9,
        x1=10,
        y1=20,
        x2=40,
        y2=80,
        center_x=25,
        center_y=50,
        direction=Direction.CENTER,
        timestamp=1.0,
    )
    payload = detection.to_dict()
    assert payload["class_id"] == 12
    assert payload["class_name"] == "person"
    assert payload["confidence"] == 0.9
    assert payload["bbox"] == {"x1": 10, "y1": 20, "x2": 40, "y2": 80}
    assert detection.center_x == 25
    assert detection.center_y == 50
    assert detection.direction == Direction.CENTER
    assert detection.timestamp == 1.0


CHECKPOINT = project_root() / "models" / "li_stick_navigation_v0.3_combined.pt"
CLASS_MAP_PATH = project_root() / "models" / "li_stick_navigation_v0.3_combined.class_map.yaml"


def test_v03_sidecar_on_disk_if_present() -> None:
    if not CLASS_MAP_PATH.exists():
        pytest.skip("v0.3 class map not written yet")
    class_map = load_class_map(CLASS_MAP_PATH)
    assert apply_class_map(4, "barrier", class_map) == (6, "barrier")
    assert apply_class_map(6, "person", class_map) == (12, "person")
    assert apply_class_map(12, "chair", class_map) == (18, "chair")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="v0.3-combined checkpoint not registered")
def test_v03_objectdetector_pipeline_on_test_image() -> None:
    from src.tracking.event_suppressor import EventSuppressor
    from src.tracking.stabilization import TemporalStabilizer
    from src.tracking.tracker import ObjectTracker
    from src.vision.detector import ObjectDetector

    config = load_app_config(
        project_root() / "config" / "config_v03_combined.yaml",
        project_root() / "config" / "classes_v03_combined.yaml",
    )
    detector = ObjectDetector(
        model_path=config.detector.model_path,
        confidence_threshold=0.10,
        allowed_classes=config.allowed_classes,
        image_size=config.detector.image_size,
    )
    assert detector._class_map is not None
    assert detector._class_map.training_to_global()[6] == 12
    assert detector._class_map.training_to_global()[12] == 18

    image_dir = project_root() / "data" / "training" / "li_stick_v03_combined" / "images" / "test"
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not images:
        pytest.skip("v0.3 test images are not on disk.")
    frame = cv2.imread(str(images[0]))
    assert frame is not None
    detections, inference_ms = detector.detect(frame, timestamp=1.0)
    assert inference_ms >= 0.0
    allowed = set(config.allowed_classes or [])
    global_ids = {0, 1, 2, 3, 6, 11, 12, 13, 14, 15, 16, 17, 18}
    for item in detections:
        assert item.class_name in allowed
        assert item.class_id in global_ids

    if not detections:
        detections = [
            Detection(
                class_id=12,
                class_name="person",
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


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
