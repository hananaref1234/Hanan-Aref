"""Build the derived v0.3-combined training view.

Combines approved Exp2b custom boxes with selected generic labels from
Roboflow Sidewalk Barriers (same-image merge first) and a filtered official
COCO chair subset. Never writes to the canonical dataset or earlier views.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.dataset.class_map import TrainingClassMap, v03_combined_class_map, write_class_map
from src.dataset.coco_generic_subset import CocoBox, load_coco_subset_boxes
from src.dataset.importers.sampling import cap_instances_by_image
from src.dataset.importers.splits import _normalized_split
from src.dataset.leakage import METADATA_FIELDS
from src.dataset.stratified_split import (
    COMBINED_V03_CLASSES,
    COMBINED_V03_MIN_TEST,
    COMBINED_V03_MIN_VAL,
    SplitRow,
    assign_group_aware_splits,
    build_groups,
    group_id_for_row,
    leakage_report,
)
from src.dataset.taxonomy import COMBINED_DETECTOR_CLASSES, CUSTOM_CLASSES, TAXONOMY_VERSION
from src.dataset.training_view import _link_or_copy, _write_dataset_yaml
from src.dataset.validator import IMAGE_SUFFIXES, SPLITS
from src.dataset.yolo_labels import parse_yolo_annotation_file

CANONICAL_DATASET = Path("data/li_stick_navigation")
ROBOFLOW_RAW = Path("data/raw/roboflow_sidewalk_barriers")
COCO_RAW = Path("data/raw/coco_2017_generic_subset")
COMBINED_VIEW = Path("data/training/li_stick_v03_combined")

CUSTOM_EXP2B_CLASSES = (
    "pole",
    "bollard",
    "stairs",
    "curb",
    "barrier",
    "tree_trunk",
)
GENERIC_CLASSES_V03 = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
)
ROBOFLOW_GENERIC_MAP = {
    "bicycle": "bicycle",
    "bus": "bus",
    "car": "car",
    "motorcycle": "motorcycle",
    "pedestrian": "person",
    "person": "person",
    "truck": "truck",
}
ROBOFLOW_CUSTOM_HAZARD_NAMES = frozenset(
    {
        "pole",
        "bollard",
        "barricade",
        "tree_trunk",
        "stairs",
        "curb",
        "stair",
        "kerb",
    }
)
GENERIC_INSTANCE_CAPS = {
    "person": 4000,
    "bicycle": 2500,
    "car": 4000,
    "motorcycle": 2500,
    "bus": 1500,
    "truck": 2000,
    "chair": 2000,
}
SAMPLE_SEED = 42


@dataclass(frozen=True)
class CombinedBox:
    sample_id: str
    filename: str
    source_dataset: str
    source_image: str
    original_class: str
    mapped_class: str
    yolo_line: str
    image_path: Path
    group_id: str
    original_source_split: str
    license: str
    conversion_method: str
    origin: str


@dataclass
class CombinedBuildResult:
    dest_root: Path
    boxes: list[CombinedBox]
    split_result: object
    stats: dict[str, object]
    partial_label: dict[str, object]
    stopped: bool = False
    stop_reason: str = ""


def load_canonical_custom_boxes(
    canonical_root: Path,
    class_map: TrainingClassMap,
) -> list[CombinedBox]:
    provenance = canonical_root / "provenance.csv"
    with provenance.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
    for raw in rows:
        mapped = str(raw.get("mapped_class", "")).strip()
        filename = str(raw.get("filename", "")).strip()
        if mapped not in CUSTOM_EXP2B_CLASSES or not filename:
            continue
        by_filename[filename].append(raw)

    global_to_training = class_map.global_to_training()
    boxes: list[CombinedBox] = []
    for filename, items in sorted(by_filename.items()):
        image_path = _canonical_image_path(canonical_root, filename)
        label_path = _canonical_label_path(canonical_root, image_path)
        first = items[0]
        source_dataset = str(first.get("source_dataset", "")).strip()
        source_image = str(first.get("source_image", "")).strip()
        group_id = group_id_for_row(source_dataset, source_image, filename)
        annotations = parse_yolo_annotation_file(
            label_path, max_class_id=len(CUSTOM_CLASSES) - 1
        )
        remaining: dict[str, list[dict[str, str]]] = defaultdict(list)
        for item in items:
            remaining[str(item.get("mapped_class", "")).strip()].append(item)
        for annotation in annotations:
            training_id = global_to_training.get(annotation.class_id)
            if training_id is None:
                continue
            mapped = COMBINED_DETECTOR_CLASSES[training_id]
            bucket = remaining.get(mapped) or []
            sample = bucket.pop(0) if bucket else first
            boxes.append(
                CombinedBox(
                    sample_id=str(sample.get("sample_id", "")).strip()
                    or f"custom::{filename}::{annotation.line_number}",
                    filename=filename,
                    source_dataset=source_dataset,
                    source_image=source_image,
                    original_class=str(sample.get("original_class", "")).strip(),
                    mapped_class=mapped,
                    yolo_line=_yolo_line(training_id, annotation),
                    image_path=image_path,
                    group_id=group_id,
                    original_source_split=_normalized_split(
                        str(sample.get("split", "")).strip()
                    ),
                    license=str(sample.get("license", "")).strip() or "CC-BY-4.0",
                    conversion_method=annotation.source_format,
                    origin="canonical_custom",
                )
            )
    return boxes


def index_roboflow_raw(raw_root: Path) -> dict[str, tuple[Path, Path, dict[int, str]]]:
    """Map image filename -> (image_path, label_path, names)."""
    names = _load_roboflow_names(raw_root)
    index: dict[str, tuple[Path, Path, dict[int, str]]] = {}
    for image_path in raw_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = _roboflow_label_for_image(image_path)
        if label_path is None:
            continue
        index[image_path.name] = (image_path, label_path, names)
    return index


def parse_roboflow_generic_boxes(
    image_path: Path,
    label_path: Path,
    names: dict[int, str],
    *,
    filename: str,
    group_id: str,
    origin: str,
    source_root: Path | None = None,
) -> tuple[list[CombinedBox], set[str], bool]:
    """Return generic boxes, source class names, and custom-hazard flag."""
    annotations = parse_yolo_annotation_file(label_path, max_class_id=max(names) if names else 0)
    source_names: set[str] = set()
    boxes: list[CombinedBox] = []
    has_custom_hazard = False
    source_image = image_path.name
    if source_root is not None:
        try:
            source_image = str(image_path.relative_to(source_root))
        except ValueError:
            source_image = image_path.name
    split = _split_from_parts(image_path)
    for annotation in annotations:
        original = names.get(annotation.class_id, "").strip()
        source_names.add(original)
        if _is_roboflow_custom_hazard(original):
            has_custom_hazard = True
        mapped = ROBOFLOW_GENERIC_MAP.get(original) or ROBOFLOW_GENERIC_MAP.get(original.lower())
        if mapped is None:
            continue
        training_id = COMBINED_DETECTOR_CLASSES.index(mapped)
        boxes.append(
            CombinedBox(
                sample_id=(
                    f"roboflow_sidewalk_barriers::{filename}::{original}::"
                    f"{annotation.line_number}"
                ),
                filename=filename,
                source_dataset="roboflow_sidewalk_barriers",
                source_image=source_image,
                original_class=original,
                mapped_class=mapped,
                yolo_line=_yolo_line(training_id, annotation),
                image_path=image_path,
                group_id=group_id,
                original_source_split=split,
                license="CC-BY-4.0",
                conversion_method=(
                    "POLYGON_TO_BOX" if annotation.source_format == "POLYGON" else "YOLO_BBOX"
                ),
                origin=origin,
            )
        )
    return boxes, source_names, has_custom_hazard


def _is_roboflow_custom_hazard(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in ROBOFLOW_CUSTOM_HAZARD_NAMES


def coco_boxes_to_combined(boxes: list[CocoBox]) -> list[CombinedBox]:
    combined: list[CombinedBox] = []
    for box in boxes:
        training_id = COMBINED_DETECTOR_CLASSES.index(box.category_name)
        filename = box.file_name
        combined.append(
            CombinedBox(
                sample_id=f"coco_2017::{box.image_id}::{box.annotation_id}",
                filename=filename,
                source_dataset="coco_2017",
                source_image=f"train2017/{box.image_id:012d}.jpg",
                original_class=box.category_name,
                mapped_class=box.category_name,
                yolo_line=(
                    f"{training_id} {box.x_center:.6f} {box.y_center:.6f} "
                    f"{box.width:.6f} {box.height:.6f}"
                ),
                image_path=box.image_path,
                group_id=f"coco_2017::{filename}",
                original_source_split="train",
                license="CC-BY-4.0-annotations_flickr_images",
                conversion_method="COCO_BBOX",
                origin="coco_chair_subset",
            )
        )
    return combined


def select_extra_roboflow_generic_images(
    index: dict[str, tuple[Path, Path, dict[int, str]]],
    already: set[str],
    current_counts: dict[str, int],
    *,
    seed: int = SAMPLE_SEED,
) -> list[CombinedBox]:
    """Add generic-only Roboflow images until caps, skipping custom-hazard files."""
    needed = [
        name
        for name, cap in GENERIC_INSTANCE_CAPS.items()
        if name != "chair" and current_counts.get(name, 0) < cap
    ]
    if not needed:
        return []

    candidates_by_class: dict[str, dict[str, list[CombinedBox]]] = {
        name: {} for name in needed
    }
    parsed_cache: dict[str, list[CombinedBox]] = {}
    for filename, (image_path, label_path, names) in index.items():
        if filename in already:
            continue
        group_id = group_id_for_row("roboflow_sidewalk_barriers", filename, filename)
        boxes, _source_names, has_custom = parse_roboflow_generic_boxes(
            image_path,
            label_path,
            names,
            filename=filename,
            group_id=group_id,
            origin="roboflow_generic_only",
            source_root=image_path.parents[2] if len(image_path.parents) >= 2 else image_path.parent,
        )
        if has_custom or not boxes:
            continue
        parsed_cache[filename] = boxes
        present = {box.mapped_class for box in boxes}
        for name in needed:
            if name in present:
                candidates_by_class[name][filename] = boxes

    selected_files: set[str] = set()
    extras: list[CombinedBox] = []
    running = dict(current_counts)
    for name in needed:
        grouped = candidates_by_class[name]
        remaining = GENERIC_INSTANCE_CAPS[name] - running.get(name, 0)
        if remaining <= 0:
            continue
        picked = cap_instances_by_image(grouped, max_instances=remaining, seed=seed)
        for box in picked:
            if box.filename in selected_files:
                continue
            selected_files.add(box.filename)
            image_boxes = parsed_cache[box.filename]
            extras.extend(image_boxes)
            for item in image_boxes:
                running[item.mapped_class] = running.get(item.mapped_class, 0) + 1
    return extras


def analyze_partial_labels(
    custom_boxes: list[CombinedBox],
    generic_boxes: list[CombinedBox],
    roboflow_index: dict[str, tuple[Path, Path, dict[int, str]]],
) -> dict[str, object]:
    custom_files = {box.filename for box in custom_boxes}
    generic_files = {box.filename for box in generic_boxes}
    rf_custom_files = {
        box.filename
        for box in custom_boxes
        if box.source_dataset == "roboflow_sidewalk_barriers"
    }
    ab_files = {
        box.filename
        for box in custom_boxes
        if box.source_dataset == "accessibility_barriers"
    }
    merged = rf_custom_files & generic_files
    rf_custom_unmerged = rf_custom_files - generic_files
    unlabeled_generic_on_rf = 0
    unlabeled_custom_on_extra = 0
    rf_custom_with_source_generic = 0
    for filename in rf_custom_files:
        entry = roboflow_index.get(filename)
        if entry is None:
            continue
        _image, label_path, names = entry
        boxes, source_names, has_custom = parse_roboflow_generic_boxes(
            entry[0],
            label_path,
            names,
            filename=filename,
            group_id="inspect",
            origin="inspect",
        )
        if boxes:
            rf_custom_with_source_generic += 1
        if filename not in generic_files and boxes:
            unlabeled_generic_on_rf += 1
        del has_custom, source_names
    extra_generic_files = {
        box.filename
        for box in generic_boxes
        if box.origin == "roboflow_generic_only"
    }
    return {
        "strategy": (
            "Same-image merge of Roboflow generic labels onto approved custom "
            "Roboflow images; extra Roboflow images only if the source file has "
            "no pole/bollard/barricade/tree_trunk/stairs/curb labels; COCO chair "
            "images keep other selected COCO generics on the same image."
        ),
        "accessibility_barriers_images": len(ab_files),
        "accessibility_barriers_unlabeled_generics": (
            "HIGH residual risk. Accessibility Barriers labels are stairs/curb "
            "only. People, chairs, and vehicles that appear in those photos are "
            "not labeled and may be treated as background."
        ),
        "roboflow_custom_images": len(rf_custom_files),
        "roboflow_custom_with_imported_generics": len(merged),
        "roboflow_custom_with_source_generic_labels": rf_custom_with_source_generic,
        "roboflow_custom_without_imported_generics": len(rf_custom_unmerged),
        "unlabeled_generic_on_included_roboflow_custom": unlabeled_generic_on_rf,
        "extra_roboflow_generic_only_images": len(extra_generic_files),
        "unlabeled_custom_on_extra_generic_images": unlabeled_custom_on_extra,
        "coco_images": len(
            {box.filename for box in generic_boxes if box.source_dataset == "coco_2017"}
        ),
        "coco_unlabeled_custom_hazards": (
            "COCO chair images do not label pole/bollard/stairs/curb/barrier/"
            "tree_trunk. Indoor chair photos make unlabeled sidewalk hazards less "
            "likely, but any outdoor COCO scene still has this risk."
        ),
        "pending_poles_on_custom_images": (
            "1,418 pole staging rows remain PENDING and are not in this dataset. "
            "Some Roboflow images may still contain unlabeled poles."
        ),
        "recommendation": (
            "Partial labels are unavoidable for Accessibility Barriers. Do not "
            "drop those stairs/curb images. Prefer the same-image Roboflow merge "
            "(applied). Do not train a second detector. Monitor person/car false "
            "positives on indoor stairs photos. Future work: ignore-regions or "
            "person labels on Accessibility Barriers."
        ),
    }


def boxes_to_split_rows(boxes: list[CombinedBox]) -> list[SplitRow]:
    rows: list[SplitRow] = []
    for box in boxes:
        rows.append(
            SplitRow(
                sample_id=box.sample_id,
                source_dataset=box.source_dataset,
                source_image=box.source_image,
                filename=box.filename,
                mapped_class=box.mapped_class,
                original_source_split=box.original_source_split,
                group_id=box.group_id,
            )
        )
    return rows


def write_combined_view(
    dest_root: Path,
    boxes: list[CombinedBox],
    filename_to_split: dict[str, str],
    class_map: TrainingClassMap,
    *,
    partial_label: dict[str, object],
    stats: dict[str, object],
) -> None:
    dest_root = dest_root.resolve()
    for split in SPLITS:
        for kind in ("images", "labels"):
            path = dest_root / kind / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    by_file: dict[str, list[CombinedBox]] = defaultdict(list)
    for box in boxes:
        by_file[box.filename].append(box)

    image_counts = {split: 0 for split in SPLITS}
    annotation_counts = {split: 0 for split in SPLITS}
    for filename, items in by_file.items():
        split = filename_to_split[filename]
        image_path = items[0].image_path
        _link_or_copy(image_path, dest_root / "images" / split / filename)
        lines = [item.yolo_line for item in items]
        (dest_root / "labels" / split / f"{Path(filename).stem}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        image_counts[split] += 1
        annotation_counts[split] += len(lines)

    _write_dataset_yaml(dest_root, class_map, class_map.training_names())
    write_class_map(dest_root / "class_map.yaml", class_map)
    _write_provenance(dest_root, boxes, filename_to_split)
    _write_metadata(dest_root, boxes, filename_to_split)
    (dest_root / "partial_label_analysis.json").write_text(
        json.dumps(partial_label, indent=2),
        encoding="utf-8",
    )
    (dest_root / "pretrain_report.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    _write_readme(dest_root, class_map, image_counts, annotation_counts, stats, partial_label)


def assemble_combined_boxes(
    *,
    canonical_root: Path,
    roboflow_root: Path,
    coco_root: Path,
    class_map: TrainingClassMap | None = None,
) -> tuple[list[CombinedBox], dict[str, object]]:
    """Load custom + generic boxes without writing a training view."""
    class_map = class_map or v03_combined_class_map()
    custom_boxes = load_canonical_custom_boxes(canonical_root, class_map)
    roboflow_index = index_roboflow_raw(roboflow_root)
    generic_boxes: list[CombinedBox] = []
    already = {box.filename for box in custom_boxes}

    for filename in sorted(already):
        entry = roboflow_index.get(filename)
        if entry is None:
            continue
        image_path, label_path, names = entry
        existing = next(box.group_id for box in custom_boxes if box.filename == filename)
        boxes, _names, _hazard = parse_roboflow_generic_boxes(
            image_path,
            label_path,
            names,
            filename=filename,
            group_id=existing,
            origin="roboflow_same_image_merge",
            source_root=roboflow_root,
        )
        generic_boxes.extend(boxes)

    counts = count_by_class(custom_boxes + generic_boxes)
    extras = select_extra_roboflow_generic_images(
        roboflow_index,
        already | {box.filename for box in generic_boxes},
        counts,
    )
    generic_boxes.extend(extras)

    coco_combined: list[CombinedBox] = []
    if (coco_root / "images").is_dir():
        coco_combined = coco_boxes_to_combined(load_coco_subset_boxes(coco_root))
        generic_boxes.extend(coco_combined)

    all_boxes = custom_boxes + generic_boxes
    sample_ids = [box.sample_id for box in all_boxes]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in combined box list.")

    partial = analyze_partial_labels(custom_boxes, generic_boxes, roboflow_index)
    stats = {
        "images": len({box.filename for box in all_boxes}),
        "annotations": len(all_boxes),
        "per_class": count_by_class(all_boxes),
        "source_contribution": source_contribution(all_boxes),
        "custom_generic_ratio": custom_generic_ratio(all_boxes),
        "origins": _count_origins(all_boxes),
        "same_image_generic_boxes": sum(
            1 for box in generic_boxes if box.origin == "roboflow_same_image_merge"
        ),
        "extra_roboflow_generic_boxes": sum(
            1 for box in generic_boxes if box.origin == "roboflow_generic_only"
        ),
        "coco_boxes": sum(1 for box in generic_boxes if box.source_dataset == "coco_2017"),
    }
    stats["partial_label"] = partial
    return all_boxes, stats


def _count_origins(boxes: list[CombinedBox]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for box in boxes:
        counts[box.origin] += 1
    return dict(counts)


def count_by_class(boxes: list[CombinedBox]) -> dict[str, int]:
    counts = {name: 0 for name in COMBINED_V03_CLASSES}
    for box in boxes:
        counts[box.mapped_class] = counts.get(box.mapped_class, 0) + 1
    return counts


def source_contribution(boxes: list[CombinedBox]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for box in boxes:
        counts[box.source_dataset] += 1
    return dict(counts)


def custom_generic_ratio(boxes: list[CombinedBox]) -> dict[str, object]:
    custom = sum(1 for box in boxes if box.mapped_class in CUSTOM_EXP2B_CLASSES)
    generic = sum(1 for box in boxes if box.mapped_class in GENERIC_CLASSES_V03)
    ratio = (generic / custom) if custom else None
    return {
        "custom_instances": custom,
        "generic_instances": generic,
        "generic_to_custom": None if ratio is None else round(ratio, 3),
        "severe_imbalance": bool(ratio is not None and ratio >= 20),
    }


def _canonical_image_path(canonical_root: Path, filename: str) -> Path:
    for split in SPLITS:
        path = canonical_root / "images" / split / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"Canonical image not found: {filename}")


def _canonical_label_path(canonical_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(canonical_root / "images")
    label_path = canonical_root / "labels" / rel.parent / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Canonical label not found: {label_path}")
    return label_path


def _load_roboflow_names(raw_root: Path) -> dict[int, str]:
    import yaml

    yaml_path = raw_root / "data.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def _roboflow_label_for_image(image_path: Path) -> Path | None:
    for parent in image_path.parents:
        if parent.name.lower() == "images":
            relative = image_path.relative_to(parent).with_suffix(".txt")
            alt = parent.parent / "labels" / relative
            if alt.is_file():
                return alt
        if parent.name.lower() in {"train", "valid", "val", "test"}:
            alt = parent / "labels" / f"{image_path.stem}.txt"
            if alt.is_file():
                return alt
    sibling = image_path.with_suffix(".txt")
    return sibling if sibling.is_file() else None


def _split_from_parts(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return ""


def _yolo_line(training_id: int, annotation) -> str:
    return (
        f"{training_id} {annotation.x_center:.6f} {annotation.y_center:.6f} "
        f"{annotation.width:.6f} {annotation.height:.6f}"
    )


def _write_provenance(
    dest_root: Path,
    boxes: list[CombinedBox],
    filename_to_split: dict[str, str],
) -> None:
    fields = (
        "sample_id",
        "source_dataset",
        "source_image",
        "original_class",
        "mapped_class",
        "license",
        "conversion_method",
        "origin",
        "split",
        "filename",
        "group_id",
        "original_source_split",
        "v03_detector_split",
        "yolo_line",
    )
    with (dest_root / "provenance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for box in boxes:
            split = filename_to_split[box.filename]
            writer.writerow(
                {
                    "sample_id": box.sample_id,
                    "source_dataset": box.source_dataset,
                    "source_image": box.source_image,
                    "original_class": box.original_class,
                    "mapped_class": box.mapped_class,
                    "license": box.license,
                    "conversion_method": box.conversion_method,
                    "origin": box.origin,
                    "split": split,
                    "filename": box.filename,
                    "group_id": box.group_id,
                    "original_source_split": box.original_source_split,
                    "v03_detector_split": split,
                    "yolo_line": box.yolo_line,
                }
            )


def _write_metadata(
    dest_root: Path,
    boxes: list[CombinedBox],
    filename_to_split: dict[str, str],
) -> None:
    by_filename: dict[str, CombinedBox] = {}
    for box in boxes:
        by_filename[box.filename] = box
    with (dest_root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for filename, box in sorted(by_filename.items()):
            split = filename_to_split[filename]
            writer.writerow(
                {
                    "filename": filename,
                    "split": split,
                    "video_id": "",
                    "session_id": "",
                    "location_id": "",
                    "sequence_id": box.group_id,
                    "frame_index": "",
                    "notes": (
                        f"v03_split={split}; origin={box.origin}; "
                        f"source={box.source_dataset}"
                    ),
                }
            )


def _write_readme(
    dest_root: Path,
    class_map: TrainingClassMap,
    images: dict[str, int],
    annotations: dict[str, int],
    stats: dict[str, object],
    partial_label: dict[str, object],
) -> None:
    rows = "\n".join(
        f"| {entry.training_id} | {entry.name} | {entry.global_id} |"
        for entry in class_map.entries
    )
    text = f"""# Training view: {class_map.model_version}

Derived Ultralytics dataset. Not the canonical taxonomy. Does not replace
`data/li_stick_navigation/`, Exp1, Exp2, or Exp2b.

## Training-specific class map

| Training ID | Name | Global Li-Stick ID |
| --- | --- | --- |
{rows}

## Counts at build time

| Split | Images | Annotations |
| --- | --- | --- |
| train | {images.get("train", 0)} | {annotations.get("train", 0)} |
| val | {images.get("val", 0)} | {annotations.get("val", 0)} |
| test | {images.get("test", 0)} | {annotations.get("test", 0)} |

See `pretrain_report.json` and `partial_label_analysis.json`.

## Partial-label strategy

{partial_label.get("strategy")}

Recommendation: {partial_label.get("recommendation")}
"""
    (dest_root / "README.md").write_text(text, encoding="utf-8")
    del stats
