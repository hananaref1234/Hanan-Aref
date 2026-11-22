"""Group-aware multilabel train/val/test assignment for derived views.

Does not write to canonical provenance. Groups keep every box from the same
source image (and Roboflow near-duplicates sharing a stem before ``.rf.``)
in one split.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.importers.splits import VALID_SPLITS, _normalized_split

DEFAULT_SPLIT_SEED = 42
DEFAULT_RATIOS = (0.8, 0.1, 0.1)

EXP2B_CLASSES = (
    "pole",
    "bollard",
    "stairs",
    "curb",
    "barrier",
    "tree_trunk",
)

EXP2B_MIN_VAL = {
    "pole": 40,
    "bollard": 40,
    "stairs": 50,
    "curb": 50,
    "barrier": 50,
    "tree_trunk": 50,
}
EXP2B_MIN_TEST = dict(EXP2B_MIN_VAL)

COMBINED_V03_CLASSES = (
    "pole",
    "bollard",
    "stairs",
    "curb",
    "barrier",
    "tree_trunk",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
)

COMBINED_V03_MIN_VAL = {
    "pole": 40,
    "bollard": 40,
    "stairs": 50,
    "curb": 50,
    "barrier": 50,
    "tree_trunk": 50,
    "person": 100,
    "bicycle": 50,
    "car": 100,
    "motorcycle": 50,
    "bus": 30,
    "truck": 50,
    "chair": 50,
}
COMBINED_V03_MIN_TEST = dict(COMBINED_V03_MIN_VAL)


@dataclass(frozen=True)
class SplitRow:
    """One promoted annotation used only to plan an Exp2b split."""

    sample_id: str
    source_dataset: str
    source_image: str
    filename: str
    mapped_class: str
    original_source_split: str
    group_id: str


@dataclass
class ImageGroup:
    group_id: str
    filenames: tuple[str, ...]
    source_images: tuple[str, ...]
    class_counts: dict[str, int]
    n_images: int
    n_boxes: int
    original_splits: tuple[str, ...]


@dataclass
class StratifiedSplitResult:
    group_to_split: dict[str, str]
    filename_to_split: dict[str, str]
    source_image_to_split: dict[str, str]
    groups: list[ImageGroup]
    class_counts: dict[str, dict[str, int]]
    image_counts: dict[str, int]
    box_counts: dict[str, int]
    leakage: dict[str, int]
    notes: list[str] = field(default_factory=list)


def roboflow_near_duplicate_key(filename: str) -> str:
    """Return the shared stem for Roboflow-augmented copies of one image."""
    name = Path(filename).name
    if ".rf." in name:
        return name.split(".rf.", 1)[0]
    return Path(name).stem


def group_id_for_row(
    source_dataset: str,
    source_image: str,
    filename: str,
) -> str:
    """Stable group key: dataset + image, collapsing Roboflow ``.rf.`` variants."""
    if source_dataset == "roboflow_sidewalk_barriers":
        return f"{source_dataset}::{roboflow_near_duplicate_key(filename)}"
    return f"{source_dataset}::{source_image or filename}"


def rows_from_canonical_provenance(path: Path) -> list[SplitRow]:
    """Load promoted canonical provenance without modifying it."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[SplitRow] = []
        for raw in reader:
            filename = str(raw.get("filename", "")).strip()
            source_image = str(raw.get("source_image", "")).strip()
            source_dataset = str(raw.get("source_dataset", "")).strip()
            mapped = str(raw.get("mapped_class", "")).strip()
            if not filename or mapped not in EXP2B_CLASSES:
                continue
            rows.append(
                SplitRow(
                    sample_id=str(raw.get("sample_id", "")).strip(),
                    source_dataset=source_dataset,
                    source_image=source_image,
                    filename=filename,
                    mapped_class=mapped,
                    original_source_split=_normalized_split(str(raw.get("split", "")).strip()),
                    group_id=group_id_for_row(source_dataset, source_image, filename),
                )
            )
        return rows


def build_groups(
    rows: list[SplitRow],
    class_names: tuple[str, ...] = EXP2B_CLASSES,
) -> list[ImageGroup]:
    by_id: dict[str, list[SplitRow]] = defaultdict(list)
    for row in rows:
        by_id[row.group_id].append(row)
    groups: list[ImageGroup] = []
    for group_id, items in sorted(by_id.items()):
        counts: dict[str, int] = {name: 0 for name in class_names}
        for item in items:
            if item.mapped_class in counts:
                counts[item.mapped_class] += 1
        groups.append(
            ImageGroup(
                group_id=group_id,
                filenames=tuple(sorted({item.filename for item in items})),
                source_images=tuple(sorted({item.source_image for item in items})),
                class_counts=counts,
                n_images=len({item.filename for item in items}),
                n_boxes=len(items),
                original_splits=tuple(
                    sorted({item.original_source_split for item in items if item.original_source_split})
                ),
            )
        )
    return groups


def assign_group_aware_splits(
    groups: list[ImageGroup],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    min_val: dict[str, int] | None = None,
    min_test: dict[str, int] | None = None,
    class_names: tuple[str, ...] = EXP2B_CLASSES,
    protect_train_share: float | None = None,
) -> StratifiedSplitResult:
    """Assign each image group to train/val/test.

    Rare classes are considered first so val/test receive meaningful counts.
    Source-image and near-duplicate groups are never split.
    """
    min_val = dict(min_val or EXP2B_MIN_VAL)
    min_test = dict(min_test or EXP2B_MIN_TEST)
    totals = {name: sum(group.class_counts.get(name, 0) for group in groups) for name in class_names}
    current = {split: {name: 0 for name in class_names} for split in VALID_SPLITS}
    image_counts = {split: 0 for split in VALID_SPLITS}
    box_counts = {split: 0 for split in VALID_SPLITS}
    assigned: dict[str, str] = {}

    ordered = _order_groups(groups, totals, class_names, seed)
    for group in ordered:
        split = _choose_split(
            group,
            current=current,
            image_counts=image_counts,
            totals=totals,
            n_images_total=sum(item.n_images for item in groups),
            ratios=ratios,
            min_val=min_val,
            min_test=min_test,
            class_names=class_names,
        )
        assigned[group.group_id] = split
        image_counts[split] += group.n_images
        box_counts[split] += group.n_boxes
        for name in class_names:
            current[split][name] += group.class_counts.get(name, 0)

    assigned, current, image_counts, box_counts, notes = _repair_minima(
        groups=groups,
        assigned=assigned,
        current=current,
        image_counts=image_counts,
        box_counts=box_counts,
        min_val=min_val,
        min_test=min_test,
        class_names=class_names,
    )
    if protect_train_share is not None:
        assigned, current, image_counts, box_counts, extra_notes = _repair_train_starvation(
            groups=groups,
            assigned=assigned,
            current=current,
            image_counts=image_counts,
            box_counts=box_counts,
            min_val=min_val,
            min_test=min_test,
            class_names=class_names,
            min_train_share=protect_train_share,
        )
        notes.extend(extra_notes)

    filename_to_split: dict[str, str] = {}
    source_image_to_split: dict[str, str] = {}
    filename_groups: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        split = assigned[group.group_id]
        for filename in group.filenames:
            previous = filename_to_split.get(filename)
            if previous and previous != split:
                raise ValueError(f"Filename {filename!r} assigned to {previous} and {split}.")
            filename_to_split[filename] = split
            filename_groups[filename].add(group.group_id)
        for source_image in group.source_images:
            previous = source_image_to_split.get(source_image)
            if previous and previous != split:
                raise ValueError(
                    f"Source image {source_image!r} assigned to {previous} and {split}."
                )
            source_image_to_split[source_image] = split

    duplicate_filenames = sum(1 for owners in filename_groups.values() if len(owners) > 1)
    group_leaks = 0
    for group in groups:
        splits = {filename_to_split[name] for name in group.filenames}
        if len(splits) != 1:
            group_leaks += 1
    by_source: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        split = assigned[group.group_id]
        for source_image in group.source_images:
            if source_image:
                by_source[source_image].add(split)
    source_image_leaks = sum(1 for splits in by_source.values() if len(splits) > 1)
    leakage = {
        "source_image_leakage": source_image_leaks,
        "filename_leakage": duplicate_filenames,
        "group_leakage": group_leaks,
        "source_overlap": source_image_leaks,
    }

    return StratifiedSplitResult(
        group_to_split=assigned,
        filename_to_split=filename_to_split,
        source_image_to_split=source_image_to_split,
        groups=groups,
        class_counts=current,
        image_counts=image_counts,
        box_counts=box_counts,
        leakage=leakage,
        notes=notes,
    )


def _order_groups(
    groups: list[ImageGroup],
    totals: dict[str, int],
    class_names: tuple[str, ...],
    seed: int,
) -> list[ImageGroup]:
    rng = random.Random(seed)
    keyed: list[tuple[int, int, str, ImageGroup]] = []
    for group in groups:
        present = [name for name in class_names if group.class_counts.get(name, 0) > 0]
        rarity = min(totals[name] for name in present) if present else 10**9
        keyed.append((rarity, -group.n_boxes, group.group_id, group))
    keyed.sort(key=lambda item: (item[0], item[1], item[2]))
    # Shuffle within equal rarity so assignment is not filename-order biased.
    buckets: dict[int, list[ImageGroup]] = defaultdict(list)
    order_keys: list[int] = []
    for rarity, _neg, _gid, group in keyed:
        if rarity not in buckets:
            order_keys.append(rarity)
        buckets[rarity].append(group)
    ordered: list[ImageGroup] = []
    for rarity in order_keys:
        bucket = buckets[rarity]
        rng.shuffle(bucket)
        ordered.extend(bucket)
    return ordered


def _choose_split(
    group: ImageGroup,
    *,
    current: dict[str, dict[str, int]],
    image_counts: dict[str, int],
    totals: dict[str, int],
    n_images_total: int,
    ratios: tuple[float, float, float],
    min_val: dict[str, int],
    min_test: dict[str, int],
    class_names: tuple[str, ...],
) -> str:
    ratio_map = {"train": ratios[0], "val": ratios[1], "test": ratios[2]}
    minima = {"train": {}, "val": min_val, "test": min_test}
    best_split = "train"
    best_score = float("-inf")
    for split in VALID_SPLITS:
        score = 0.0
        for name in class_names:
            n = group.class_counts.get(name, 0)
            if n <= 0:
                continue
            needed = minima[split].get(name, 0) - current[split][name]
            if needed > 0:
                score += 10_000.0 + min(n, needed) * 100.0
            target = ratio_map[split] * max(totals[name], 1)
            remaining = target - current[split][name]
            score += remaining
        target_images = ratio_map[split] * max(n_images_total, 1)
        score += (target_images - image_counts[split]) * 0.25
        if score > best_score:
            best_score = score
            best_split = split
    return best_split


def _repair_minima(
    *,
    groups: list[ImageGroup],
    assigned: dict[str, str],
    current: dict[str, dict[str, int]],
    image_counts: dict[str, int],
    box_counts: dict[str, int],
    min_val: dict[str, int],
    min_test: dict[str, int],
    class_names: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, int], dict[str, int], list[str]]:
    """Move smallest train groups into val/test if a class is still short."""
    by_id = {group.group_id: group for group in groups}
    notes: list[str] = []

    def _move(group: ImageGroup, dest: str) -> None:
        src = assigned[group.group_id]
        if src == dest:
            return
        assigned[group.group_id] = dest
        image_counts[src] -= group.n_images
        image_counts[dest] += group.n_images
        box_counts[src] -= group.n_boxes
        box_counts[dest] += group.n_boxes
        for name in class_names:
            n = group.class_counts.get(name, 0)
            current[src][name] -= n
            current[dest][name] += n

    for dest, minima in (("val", min_val), ("test", min_test)):
        for name, needed in minima.items():
            if current[dest][name] >= needed:
                continue
            candidates = [
                by_id[gid]
                for gid, split in assigned.items()
                if split == "train" and by_id[gid].class_counts.get(name, 0) > 0
            ]
            candidates.sort(key=lambda group: (group.class_counts.get(name, 0), group.n_boxes, group.group_id))
            for group in candidates:
                if current[dest][name] >= needed:
                    break
                _move(group, dest)
                notes.append(
                    f"Moved group {group.group_id} train->{dest} to cover {name}."
                )
            if current[dest][name] < needed:
                notes.append(
                    f"Could not reach {dest} target {needed} for {name}; "
                    f"have {current[dest][name]}."
                )
    return assigned, current, image_counts, box_counts, notes


def _repair_train_starvation(
    *,
    groups: list[ImageGroup],
    assigned: dict[str, str],
    current: dict[str, dict[str, int]],
    image_counts: dict[str, int],
    box_counts: dict[str, int],
    min_val: dict[str, int],
    min_test: dict[str, int],
    class_names: tuple[str, ...],
    min_train_share: float,
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, int], dict[str, int], list[str]]:
    """Move surplus val/test groups back to train if a class is train-starved.

    Does not go below val/test minima. Used when rare-class coverage would
    otherwise invert the split (more val than train).
    """
    by_id = {group.group_id: group for group in groups}
    notes: list[str] = []
    totals = {
        name: current["train"][name] + current["val"][name] + current["test"][name]
        for name in class_names
    }

    def _move(group: ImageGroup, dest: str) -> None:
        src = assigned[group.group_id]
        if src == dest:
            return
        assigned[group.group_id] = dest
        image_counts[src] -= group.n_images
        image_counts[dest] += group.n_images
        box_counts[src] -= group.n_boxes
        box_counts[dest] += group.n_boxes
        for name in class_names:
            n = group.class_counts.get(name, 0)
            current[src][name] -= n
            current[dest][name] += n

    for name in class_names:
        total = max(totals[name], 1)
        target_train = int(total * min_train_share)
        if current["train"][name] >= target_train:
            continue
        for src, _minima in (("val", min_val), ("test", min_test)):
            candidates = [
                by_id[gid]
                for gid, split in assigned.items()
                if split == src and by_id[gid].class_counts.get(name, 0) > 0
            ]
            candidates.sort(
                key=lambda group: (-group.class_counts.get(name, 0), -group.n_boxes, group.group_id)
            )
            for group in candidates:
                if current["train"][name] >= target_train:
                    break
                if _move_would_break_minima(
                    group,
                    src,
                    current=current,
                    min_val=min_val,
                    min_test=min_test,
                    class_names=class_names,
                ):
                    continue
                _move(group, "train")
                notes.append(
                    f"Moved group {group.group_id} {src}->train to restore {name} train share."
                )
        if current["train"][name] < target_train:
            notes.append(
                f"Could not reach train share {min_train_share:.0%} for {name}; "
                f"have {current['train'][name]}/{total}."
            )
    return assigned, current, image_counts, box_counts, notes


def _move_would_break_minima(
    group: ImageGroup,
    src: str,
    *,
    current: dict[str, dict[str, int]],
    min_val: dict[str, int],
    min_test: dict[str, int],
    class_names: tuple[str, ...],
) -> bool:
    floors = {"val": min_val, "test": min_test}.get(src, {})
    for name in class_names:
        floor = floors.get(name, 0)
        if current[src][name] - group.class_counts.get(name, 0) < floor:
            return True
    return False


def write_split_assignment(
    path: Path,
    rows: list[SplitRow],
    result: StratifiedSplitResult,
    *,
    split_column: str = "exp2b_detector_split",
) -> None:
    """Write an auditable per-box assignment table for the derived view."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "sample_id",
        "source_dataset",
        "source_image",
        "filename",
        "mapped_class",
        "group_id",
        "original_source_split",
        split_column,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "source_dataset": row.source_dataset,
                    "source_image": row.source_image,
                    "filename": row.filename,
                    "mapped_class": row.mapped_class,
                    "group_id": row.group_id,
                    "original_source_split": row.original_source_split,
                    split_column: result.filename_to_split[row.filename],
                }
            )


def leakage_report(result: StratifiedSplitResult) -> dict[str, int]:
    """Return leakage counters used by the Exp2b pre-train gate."""
    return {
        "source_overlap": result.leakage["source_overlap"],
        "source_image_leakage": result.leakage["source_image_leakage"],
        "exact_duplicate_leakage": result.leakage["filename_leakage"] + result.leakage["group_leakage"],
        "group_leakage": result.leakage["group_leakage"],
    }
