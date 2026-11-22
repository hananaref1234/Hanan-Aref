"""Per-split class counts for a YOLO dataset or training view."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.dataset.class_map import TrainingClassMap
from src.dataset.taxonomy import CLASS_TO_ID, CUSTOM_CLASSES
from src.dataset.validator import IMAGE_SUFFIXES, SPLITS
from src.dataset.yolo_labels import parse_yolo_label_file


def count_split_classes(
    dataset_root: Path,
    *,
    class_names: dict[int, str] | None = None,
    class_map: TrainingClassMap | None = None,
) -> dict[str, object]:
    """Count images, annotations, and per-class instances per split."""
    dataset_root = dataset_root.resolve()
    if class_names is None:
        from src.dataset.taxonomy import load_dataset_yaml_names

        class_names = load_dataset_yaml_names(dataset_root / "dataset.yaml")

    training_to_global = class_map.training_to_global() if class_map else {}
    if class_map is not None:
        global_names = class_map.global_names()
    else:
        global_names = {gid: name for name, gid in CLASS_TO_ID.items()}

    images: dict[str, int] = {split: 0 for split in SPLITS}
    annotations: dict[str, int] = {split: 0 for split in SPLITS}
    per_class: dict[str, dict[str, int]] = {
        split: {name: 0 for name in class_names.values()} for split in SPLITS
    }

    max_class_id = max(class_names) if class_names else max(CLASS_TO_ID.values())
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            boxes = parse_yolo_label_file(label_path, max_class_id=max_class_id)
            if not boxes:
                continue
            images[split] += 1
            annotations[split] += len(boxes)
            for box in boxes:
                training_id = box.class_id
                if training_to_global:
                    global_id = training_to_global.get(training_id, training_id)
                    name = global_names.get(global_id, str(global_id))
                else:
                    name = class_names.get(training_id, str(training_id))
                if name in per_class[split]:
                    per_class[split][name] += 1

    totals = {
        name: sum(per_class[split].get(name, 0) for split in SPLITS)
        for name in sorted({name for split in per_class for name in per_class[split]})
    }
    return {
        "images": images,
        "annotations": annotations,
        "per_class_by_split": per_class,
        "per_class_total": totals,
    }


def format_balance_report(
    counts: dict[str, object],
    *,
    classes: tuple[str, ...] = tuple(CUSTOM_CLASSES),
) -> str:
    """Render a human-readable class-balance table."""
    totals = counts.get("per_class_total", {})
    if not isinstance(totals, dict):
        totals = {}
    trained = [name for name in classes if totals.get(name, 0) > 0]
    lines = [
        "Class balance (instances in derived/canonical labels):",
        "",
        "| Class | Total | train | val | test |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    per_split = counts.get("per_class_by_split", {})
    for name in trained:
        row = totals.get(name, 0)
        train = val = test = 0
        if isinstance(per_split, dict):
            train = per_split.get("train", {}).get(name, 0) if isinstance(per_split.get("train"), dict) else 0
            val = per_split.get("val", {}).get(name, 0) if isinstance(per_split.get("val"), dict) else 0
            test = per_split.get("test", {}).get(name, 0) if isinstance(per_split.get("test"), dict) else 0
        lines.append(f"| {name} | {row} | {train} | {val} | {test} |")
    images = counts.get("images", {})
    annotations = counts.get("annotations", {})
    if isinstance(images, dict) and isinstance(annotations, dict):
        lines.extend(
            [
                "",
                f"Images: train={images.get('train', 0)} val={images.get('val', 0)} test={images.get('test', 0)}",
                f"Annotations: train={annotations.get('train', 0)} val={annotations.get('val', 0)} test={annotations.get('test', 0)}",
            ]
        )
    return "\n".join(lines)
