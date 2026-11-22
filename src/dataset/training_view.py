"""Build a derived YOLO training view without modifying the canonical dataset.

Ultralytics needs contiguous class IDs starting at 0. Taxonomy v0.2 keeps
global IDs unchanged (stairs=2, curb=3, barrier=6, tree_trunk=11, …). This
module hardlinks/copies images and rewrites labels into a separate directory
with experiment-specific training IDs.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.dataset.class_map import TrainingClassMap, write_class_map
from src.dataset.taxonomy import CUSTOM_CLASSES, TAXONOMY_VERSION
from src.dataset.validator import IMAGE_SUFFIXES, SPLITS

CANONICAL_DATASET = Path("data/li_stick_navigation")
EXP1_TRAINING_VIEW = Path("data/training/li_stick_v02_exp1")
EXP2_TRAINING_VIEW = Path("data/training/li_stick_v02_exp2")
EXP2B_TRAINING_VIEW = Path("data/training/li_stick_v02_exp2b")
EXP3_COMBINED_TRAINING_VIEW = Path("data/training/li_stick_v03_combined")


@dataclass(frozen=True)
class TrainingViewStats:
    dest_root: Path
    images: dict[str, int]
    annotations: dict[str, int]
    skipped_unmapped: int
    linked_images: int
    copied_images: int


def remap_label_text(text: str, global_to_training: dict[int, int]) -> tuple[str, int]:
    """Rewrite YOLO label lines from global IDs to training IDs.

    Lines whose class is not in the map are dropped (not turned into fakes).
    Returns the remapped text and the number of dropped boxes.
    """
    out_lines: list[str] = []
    skipped = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Malformed YOLO label line (expected 5 fields): {line!r}"
            )
        try:
            global_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"Non-integer class id in label line: {line!r}") from exc
        training_id = global_to_training.get(global_id)
        if training_id is None:
            skipped += 1
            continue
        parts[0] = str(training_id)
        out_lines.append(" ".join(parts))
    remapped = "\n".join(out_lines)
    if remapped:
        remapped += "\n"
    return remapped, skipped


def _link_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size == src.stat().st_size:
            return "exists"
        dest.unlink()
    try:
        os.link(src, dest)
        return "link"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def build_training_view(
    source_root: Path,
    dest_root: Path,
    class_map: TrainingClassMap,
    *,
    split_by_filename: dict[str, str] | None = None,
    copy_sidecars: bool = True,
) -> TrainingViewStats:
    """Create a derived dataset with remapped labels. Never writes to source_root.

    If ``split_by_filename`` is set, images are gathered from every canonical
    split and written into the named dest split. Canonical folders are not
    modified.
    """
    source_root = source_root.resolve()
    dest_root = dest_root.resolve()
    if dest_root == source_root:
        raise ValueError("Training view destination must not be the canonical dataset.")
    if source_root in dest_root.parents:
        raise ValueError("Training view destination must not be inside the canonical dataset.")

    yaml_src = source_root / "dataset.yaml"
    if not yaml_src.exists():
        raise FileNotFoundError(f"Canonical dataset.yaml not found: {yaml_src}")

    global_to_training = class_map.global_to_training()
    names = class_map.training_names()
    image_counts = {split: 0 for split in SPLITS}
    annotation_counts = {split: 0 for split in SPLITS}
    skipped_unmapped = 0
    linked_images = 0
    copied_images = 0

    for split in SPLITS:
        dest_image_dir = dest_root / "images" / split
        dest_label_dir = dest_root / "labels" / split
        dest_image_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)

    planned = _iter_source_images(source_root, split_by_filename)
    for image_path, dest_split in planned:
        if dest_split not in SPLITS:
            raise ValueError(f"Invalid dest split {dest_split!r} for {image_path.name}.")
        label_path = _label_for_image(source_root, image_path)
        remapped, skipped = remap_label_text(
            label_path.read_text(encoding="utf-8"),
            global_to_training,
        )
        skipped_unmapped += skipped
        dest_image_dir = dest_root / "images" / dest_split
        dest_label_dir = dest_root / "labels" / dest_split
        dest_label = dest_label_dir / f"{image_path.stem}.txt"
        dest_label.write_text(remapped, encoding="utf-8")
        method = _link_or_copy(image_path, dest_image_dir / image_path.name)
        if method == "link":
            linked_images += 1
        elif method == "copy":
            copied_images += 1
        image_counts[dest_split] += 1
        if remapped.strip():
            annotation_counts[dest_split] += sum(
                1 for line in remapped.splitlines() if line.strip()
            )

    _write_dataset_yaml(dest_root, class_map, names)
    write_class_map(dest_root / "class_map.yaml", class_map)
    if copy_sidecars:
        _copy_sidecar_tables(source_root, dest_root)
    _write_readme(dest_root, source_root, class_map, image_counts, annotation_counts)

    return TrainingViewStats(
        dest_root=dest_root,
        images=image_counts,
        annotations=annotation_counts,
        skipped_unmapped=skipped_unmapped,
        linked_images=linked_images,
        copied_images=copied_images,
    )


def _iter_source_images(
    source_root: Path,
    split_by_filename: dict[str, str] | None,
) -> list[tuple[Path, str]]:
    planned: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for split in SPLITS:
        image_dir = source_root / "images" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            dest_split = split
            if split_by_filename is not None:
                dest_split = split_by_filename.get(image_path.name)
                if dest_split is None:
                    continue
            if image_path.name in seen:
                raise ValueError(f"Duplicate source filename {image_path.name!r}.")
            seen.add(image_path.name)
            planned.append((image_path, dest_split))
    if split_by_filename is not None:
        missing = sorted(set(split_by_filename) - seen)
        if missing:
            raise FileNotFoundError(
                f"Split plan references missing canonical images: {missing[:5]}"
            )
    return planned


def _label_for_image(source_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(source_root / "images")
    label_path = source_root / "labels" / rel.parent / f"{image_path.stem}.txt"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label for {image_path.relative_to(source_root)}. "
            "Refusing to build an incomplete training view."
        )
    return label_path


def _write_dataset_yaml(
    dest_root: Path,
    class_map: TrainingClassMap,
    names: dict[int, str],
) -> None:
    path_str = dest_root.as_posix()
    name_lines = "\n".join(f"  {index}: {name}" for index, name in sorted(names.items()))
    text = (
        "# Derived Ultralytics training view -- do not treat as the canonical taxonomy.\n"
        f"# Canonical dataset: data/li_stick_navigation/ (taxonomy {TAXONOMY_VERSION})\n"
        f"# Model version: {class_map.model_version}\n"
        "# Training IDs are remapped; see class_map.yaml.\n"
        "\n"
        f"path: {path_str}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        f"taxonomy_version: {class_map.taxonomy_version}\n"
        f"dataset_version: {class_map.dataset_version}\n"
        f"training_view: {class_map.model_version}\n"
        "\n"
        "names:\n"
        f"{name_lines}\n"
    )
    (dest_root / "dataset.yaml").write_text(text, encoding="utf-8")


def _copy_sidecar_tables(source_root: Path, dest_root: Path) -> None:
    for name in ("metadata.csv", "provenance.csv"):
        src = source_root / name
        if src.exists():
            shutil.copy2(src, dest_root / name)


def _write_readme(
    dest_root: Path,
    source_root: Path,
    class_map: TrainingClassMap,
    images: dict[str, int],
    annotations: dict[str, int],
) -> None:
    rows = "\n".join(
        f"| {entry.training_id} | {entry.name} | {entry.global_id} |"
        for entry in class_map.entries
    )
    trained = {entry.name for entry in class_map.entries}
    omitted = [name for name in CUSTOM_CLASSES if name not in trained]
    omitted_text = ", ".join(omitted) if omitted else "none"
    text = f"""# Training view: {class_map.model_version}

This directory is a **derived** Ultralytics dataset. It is not the canonical
Li-Stick taxonomy and must not replace `data/li_stick_navigation/`.

## Provenance

- Canonical dataset: `{source_root.as_posix()}`
- Dataset version: `{class_map.dataset_version}`
- Taxonomy version: `{class_map.taxonomy_version}` (unchanged)
- Images are hardlinked or copied; labels are rewritten with training IDs.
- `metadata.csv` and `provenance.csv` are copies of the canonical sidecars.

## Why remap?

Ultralytics trains a head with contiguous class IDs starting at 0. The
project-wide taxonomy still uses the frozen v0.2 global IDs (0–11). Training
this experiment on the sparse 12-class YAML would create empty classes with
no labels. Empty classes are not invented here.

## Training-specific class map

| Training ID | Name | Global Li-Stick ID |
| --- | --- | --- |
{rows}

See `class_map.yaml`. The registered checkpoint sidecar
`models/<checkpoint>.class_map.yaml` uses the same mapping so
`ObjectDetector` emits global IDs and Li-Stick names.

## Classes not in this experiment

{omitted_text}

## Counts at build time

| Split | Images | Annotations |
| --- | --- | --- |
| train | {images.get("train", 0)} | {annotations.get("train", 0)} |
| val | {images.get("val", 0)} | {annotations.get("val", 0)} |
| test | {images.get("test", 0)} | {annotations.get("test", 0)} |

## Limitations

- Custom hazard classes only. No COCO/generic classes in this checkpoint.
- This experiment does not learn up/down. Direction is not a detector label.
- Not a safety-certified model.
"""
    (dest_root / "README.md").write_text(text, encoding="utf-8")
