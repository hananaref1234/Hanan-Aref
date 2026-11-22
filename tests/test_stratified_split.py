"""Group-aware multilabel split tests for Exp2b."""

from __future__ import annotations

from pathlib import Path

from src.dataset.class_map import apply_class_map, exp2b_six_class_class_map
from src.dataset.stratified_split import (
    ImageGroup,
    SplitRow,
    assign_group_aware_splits,
    build_groups,
    group_id_for_row,
    leakage_report,
    roboflow_near_duplicate_key,
    write_split_assignment,
)
from src.training.config import load_training_config
from src.utils.config_loader import load_app_config
from src.utils.paths import project_root


def _group(
    group_id: str,
    *,
    filenames: tuple[str, ...],
    class_counts: dict[str, int],
    source_images: tuple[str, ...] | None = None,
) -> ImageGroup:
    n_boxes = sum(class_counts.values())
    return ImageGroup(
        group_id=group_id,
        filenames=filenames,
        source_images=source_images or filenames,
        class_counts={
            "pole": 0,
            "bollard": 0,
            "stairs": 0,
            "curb": 0,
            "barrier": 0,
            "tree_trunk": 0,
            **class_counts,
        },
        n_images=len(filenames),
        n_boxes=n_boxes,
        original_splits=("train",),
    )


def test_roboflow_near_duplicates_share_group_id() -> None:
    a = "frame_0001_jpg.rf.aaa.jpg"
    b = "frame_0001_jpg.rf.bbb.jpg"
    assert roboflow_near_duplicate_key(a) == roboflow_near_duplicate_key(b)
    gid_a = group_id_for_row("roboflow_sidewalk_barriers", "train/images/" + a, a)
    gid_b = group_id_for_row("roboflow_sidewalk_barriers", "val/images/" + b, b)
    assert gid_a == gid_b
    other = group_id_for_row(
        "accessibility_barriers", "images/0001.jpg", "0001.jpg"
    )
    assert other != gid_a


def test_split_is_deterministic_and_group_safe() -> None:
    groups = []
    for index in range(40):
        groups.append(
            _group(
                f"pole-{index:02d}",
                filenames=(f"pole_{index}.jpg",),
                class_counts={"pole": 2},
            )
        )
        groups.append(
            _group(
                f"bollard-{index:02d}",
                filenames=(f"bollard_{index}.jpg", f"bollard_{index}b.jpg")
                if index < 5
                else (f"bollard_{index}.jpg",),
                class_counts={"bollard": 3},
            )
        )
        groups.append(
            _group(
                f"stairs-{index:02d}",
                filenames=(f"stairs_{index}.jpg",),
                class_counts={"stairs": 1},
            )
        )
        groups.append(
            _group(
                f"curb-{index:02d}",
                filenames=(f"curb_{index}.jpg",),
                class_counts={"curb": 1},
            )
        )
        groups.append(
            _group(
                f"barrier-{index:02d}",
                filenames=(f"barrier_{index}.jpg",),
                class_counts={"barrier": 2},
            )
        )
        groups.append(
            _group(
                f"tree-{index:02d}",
                filenames=(f"tree_{index}.jpg",),
                class_counts={"tree_trunk": 2},
            )
        )
    first = assign_group_aware_splits(groups, seed=42)
    second = assign_group_aware_splits(groups, seed=42)
    assert first.group_to_split == second.group_to_split
    leaks = leakage_report(first)
    assert leaks["source_image_leakage"] == 0
    assert leaks["exact_duplicate_leakage"] == 0
    for name in ("pole", "bollard", "stairs", "curb", "barrier", "tree_trunk"):
        assert first.class_counts["val"][name] > 0
        assert first.class_counts["test"][name] > 0
    grouped = [group for group in groups if len(group.filenames) > 1]
    for group in grouped:
        splits = {first.filename_to_split[name] for name in group.filenames}
        assert len(splits) == 1


def test_original_source_split_is_preserved_in_assignment(tmp_path: Path) -> None:
    rows = [
        SplitRow(
            sample_id="s1",
            source_dataset="roboflow_sidewalk_barriers",
            source_image="train/images/a_jpg.rf.111.jpg",
            filename="a_jpg.rf.111.jpg",
            mapped_class="pole",
            original_source_split="train",
            group_id=group_id_for_row(
                "roboflow_sidewalk_barriers",
                "train/images/a_jpg.rf.111.jpg",
                "a_jpg.rf.111.jpg",
            ),
        ),
        SplitRow(
            sample_id="s2",
            source_dataset="roboflow_sidewalk_barriers",
            source_image="test/images/a_jpg.rf.222.jpg",
            filename="a_jpg.rf.222.jpg",
            mapped_class="pole",
            original_source_split="test",
            group_id=group_id_for_row(
                "roboflow_sidewalk_barriers",
                "test/images/a_jpg.rf.222.jpg",
                "a_jpg.rf.222.jpg",
            ),
        ),
        SplitRow(
            sample_id="s3",
            source_dataset="accessibility_barriers",
            source_image="images/b.jpg",
            filename="b.jpg",
            mapped_class="stairs",
            original_source_split="val",
            group_id=group_id_for_row(
                "accessibility_barriers", "images/b.jpg", "b.jpg"
            ),
        ),
    ]
    groups = build_groups(rows)
    assert len(groups) == 2
    result = assign_group_aware_splits(
        groups,
        seed=1,
        min_val={
            name: 0
            for name in (
                "pole",
                "bollard",
                "stairs",
                "curb",
                "barrier",
                "tree_trunk",
            )
        },
        min_test={
            name: 0
            for name in (
                "pole",
                "bollard",
                "stairs",
                "curb",
                "barrier",
                "tree_trunk",
            )
        },
    )
    path = tmp_path / "split_assignment.csv"
    write_split_assignment(path, rows, result)
    text = path.read_text(encoding="utf-8")
    assert "original_source_split" in text
    assert "exp2b_detector_split" in text
    rf_split = result.filename_to_split["a_jpg.rf.111.jpg"]
    assert result.filename_to_split["a_jpg.rf.222.jpg"] == rf_split


def test_exp2b_class_map_matches_exp2_ids() -> None:
    class_map = exp2b_six_class_class_map()
    assert class_map.model_version == "li-stick-navigation-v0.2-exp2b"
    assert class_map.training_to_global() == {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 11}
    assert apply_class_map(4, "barrier", class_map) == (6, "barrier")
    assert apply_class_map(5, "tree_trunk", class_map) == (11, "tree_trunk")


def test_exp2b_configs_are_selectable() -> None:
    training = load_training_config(project_root() / "config" / "training_v02_exp2b.yaml")
    assert training.name == "v02_exp2b"
    assert training.register == project_root() / "models" / "li_stick_navigation_v0.2_exp2b.pt"
    assert training.base_model == "yolo26n.pt"
    assert training.epochs == 100
    assert "li_stick_v02_exp2b" in str(training.dataset_yaml)
    config = load_app_config(
        project_root() / "config" / "config_v02_exp2b.yaml",
        project_root() / "config" / "classes_v02_exp2b.yaml",
    )
    assert config.detector.model_path == "models/li_stick_navigation_v0.2_exp2b.pt"
    assert config.allowed_classes == [
        "pole",
        "bollard",
        "stairs",
        "curb",
        "barrier",
        "tree_trunk",
    ]
