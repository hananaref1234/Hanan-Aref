"""Training-view remap and class-map tests (no YOLO training)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset.class_map import (
    ClassMapError,
    apply_class_map,
    exp1_stairs_curb_class_map,
    exp2_six_class_class_map,
    load_class_map,
    sidecar_path_for_checkpoint,
    write_class_map,
)
from src.dataset.taxonomy import CLASS_TO_ID, CUSTOM_CLASSES
from src.dataset.training_view import build_training_view, remap_label_text
from src.dataset.validator import validate_dataset
from src.training.config import load_training_config
from src.utils.config_loader import load_app_config
from src.utils.paths import project_root


def test_remap_stairs_curb_global_ids() -> None:
    text = "2 0.50 0.50 0.20 0.20\n3 0.10 0.20 0.10 0.10\n"
    remapped, skipped = remap_label_text(text, {2: 0, 3: 1})
    lines = remapped.strip().splitlines()
    assert skipped == 0
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("1 ")
    assert "0.50 0.50 0.20 0.20" in lines[0]


def test_remap_drops_unmapped_classes_without_fakes() -> None:
    text = "0 0.50 0.50 0.20 0.20\n2 0.10 0.20 0.10 0.10\n7 0.30 0.30 0.05 0.05\n"
    remapped, skipped = remap_label_text(text, {2: 0, 3: 1})
    assert skipped == 2
    assert remapped.strip() == "0 0.10 0.20 0.10 0.10"
    assert "0.50" not in remapped


def test_exp1_class_map_preserves_global_taxonomy() -> None:
    class_map = exp1_stairs_curb_class_map()
    assert class_map.training_names() == {0: "stairs", 1: "curb"}
    assert class_map.training_to_global() == {0: 2, 1: 3}
    assert class_map.global_to_training() == {2: 0, 3: 1}
    assert CLASS_TO_ID["stairs"] == 2
    assert CLASS_TO_ID["curb"] == 3
    assert CUSTOM_CLASSES[2] == "stairs"
    assert len(CUSTOM_CLASSES) == 12


def test_exp2_class_map_preserves_global_taxonomy() -> None:
    class_map = exp2_six_class_class_map()
    assert class_map.training_names() == {
        0: "pole",
        1: "bollard",
        2: "stairs",
        3: "curb",
        4: "barrier",
        5: "tree_trunk",
    }
    assert class_map.training_to_global() == {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 11}
    class_id, name = apply_class_map(4, "barrier", class_map)
    assert class_id == 6
    assert name == "barrier"
    class_id, name = apply_class_map(5, "tree_trunk", class_map)
    assert class_id == 11
    assert name == "tree_trunk"


def test_class_map_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "class_map.yaml"
    write_class_map(path, exp1_stairs_curb_class_map())
    loaded = load_class_map(path)
    assert loaded.training_to_global() == {0: 2, 1: 3}
    class_id, name = apply_class_map(0, "stairs", loaded)
    assert class_id == 2
    assert name == "stairs"
    class_id, name = apply_class_map(1, "curb", loaded)
    assert class_id == 3
    assert name == "curb"


def test_apply_class_map_passthrough_without_sidecar() -> None:
    class_id, name = apply_class_map(0, "person", None)
    assert class_id == 0
    assert name == "person"


def test_sidecar_path_for_checkpoint() -> None:
    path = sidecar_path_for_checkpoint(Path("models/li_stick_navigation_v0.2_exp1.pt"))
    assert path == Path("models/li_stick_navigation_v0.2_exp1.class_map.yaml")


def test_class_map_rejects_wrong_global_id(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "training_to_global:\n  0:\n    name: stairs\n    global_id: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ClassMapError, match="global id 2"):
        load_class_map(path)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_build_training_view_does_not_modify_source(tmp_path: Path) -> None:
    source = tmp_path / "canonical"
    dest = tmp_path / "view"
    for split in ("train", "val", "test"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    (source / "dataset.yaml").write_text("names:\n  0: pole\n", encoding="utf-8")
    _write_png(source / "images" / "train" / "a.png")
    original = "2 0.5 0.5 0.2 0.2\n3 0.2 0.2 0.1 0.1\n"
    (source / "labels" / "train" / "a.txt").write_text(original, encoding="utf-8")

    stats = build_training_view(source, dest, exp1_stairs_curb_class_map())
    assert stats.images["train"] == 1
    assert stats.annotations["train"] == 2
    assert (source / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == original
    remapped = (dest / "labels" / "train" / "a.txt").read_text(encoding="utf-8")
    assert remapped.splitlines()[0].startswith("0 ")
    assert remapped.splitlines()[1].startswith("1 ")
    yaml_text = (dest / "dataset.yaml").read_text(encoding="utf-8")
    assert "training_view:" in yaml_text
    assert "stairs" in yaml_text
    assert "pole" not in yaml_text
    assert dest.as_posix() in yaml_text.replace("\\", "/")


def test_validator_accepts_training_view(tmp_path: Path) -> None:
    root = tmp_path / "view"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    (root / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "taxonomy_version: v0.2\ntraining_view: li-stick-navigation-v0.2-exp1\n"
        "names:\n  0: stairs\n  1: curb\n",
        encoding="utf-8",
    )
    _write_png(root / "images" / "train" / "a.png")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    report = validate_dataset(root)
    assert report.passed
    assert report.annotations_total == 1


def test_validator_rejects_sparse_view_without_flag(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    (root / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "taxonomy_version: v0.2\n"
        "names:\n  0: stairs\n  1: curb\n",
        encoding="utf-8",
    )
    _write_png(root / "images" / "train" / "a.png")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    report = validate_dataset(root)
    assert report.passed is False
    assert any("frozen custom taxonomy" in issue.message for issue in report.issues)


def test_default_training_config_still_canonical() -> None:
    config = load_training_config()
    assert config.dataset_yaml == project_root() / "data" / "li_stick_navigation" / "dataset.yaml"
    assert config.register is None


def test_exp1_training_config_is_selectable() -> None:
    config = load_training_config(project_root() / "config" / "training_v02_exp1.yaml")
    assert config.base_model == "yolo26n.pt"
    assert config.epochs == 80
    assert config.image_size == 640
    assert config.patience == 20
    assert config.augmentation.flipud == 0.0
    assert config.augmentation.fliplr == 0.5
    assert "li_stick_v02_exp1" in str(config.dataset_yaml)
    assert config.register == project_root() / "models" / "li_stick_navigation_v0.2_exp1.pt"
    assert config.name == "v02_exp1"


def test_exp1_webcam_config_is_selectable() -> None:
    config = load_app_config(
        project_root() / "config" / "config_v02_exp1.yaml",
        project_root() / "config" / "classes_v02_exp1.yaml",
    )
    assert config.detector.model_path == "models/li_stick_navigation_v0.2_exp1.pt"
    assert config.allowed_classes == ["stairs", "curb"]


def test_exp2_training_config_is_selectable() -> None:
    config = load_training_config(project_root() / "config" / "training_v02_exp2.yaml")
    assert config.base_model == "yolo26n.pt"
    assert config.epochs == 100
    assert config.patience == 20
    assert "li_stick_v02_exp2" in str(config.dataset_yaml)
    assert config.register == project_root() / "models" / "li_stick_navigation_v0.2_exp2.pt"
    assert config.name == "v02_exp2"


def test_exp2_webcam_config_is_selectable() -> None:
    config = load_app_config(
        project_root() / "config" / "config_v02_exp2.yaml",
        project_root() / "config" / "classes_v02_exp2.yaml",
    )
    assert config.detector.model_path == "models/li_stick_navigation_v0.2_exp2.pt"
    assert config.allowed_classes == [
        "pole",
        "bollard",
        "stairs",
        "curb",
        "barrier",
        "tree_trunk",
    ]


def test_production_webcam_config_still_pretrained() -> None:
    config = load_app_config()
    assert config.detector.model_path == "yolo26n.pt"
    assert config.allowed_classes is not None
    assert "person" in config.allowed_classes
    assert config.allowed_classes != ["stairs", "curb"]


def test_build_training_view_can_reassign_splits(tmp_path: Path) -> None:
    source = tmp_path / "canonical"
    dest = tmp_path / "view"
    for split in ("train", "val", "test"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    (source / "dataset.yaml").write_text("names:\n  0: pole\n", encoding="utf-8")
    _write_png(source / "images" / "train" / "a.png")
    (source / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    stats = build_training_view(
        source,
        dest,
        exp2_six_class_class_map(),
        split_by_filename={"a.png": "val"},
        copy_sidecars=False,
    )
    assert stats.images["val"] == 1
    assert stats.images["train"] == 0
    assert (dest / "images" / "val" / "a.png").is_file()
    assert not (dest / "images" / "train" / "a.png").exists()
