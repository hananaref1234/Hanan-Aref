"""Promote reviewed staging samples into the frozen YOLO training folders.

Never randomly split a sequence. Preserve source splits when they are
train/val/test. Unsplit sources are assigned by unique source_image with a
seeded 80/10/10 split. Do not copy PENDING, REJECTED, or AMBIGUOUS samples.
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from src.dataset.importers.boxes import PixelBox, xyxy_to_yolo
from src.dataset.importers.provenance import ProvenanceRecord, read_provenance, write_provenance
from src.dataset.importers.review import class_for_promotion
from src.dataset.importers.splits import (
    DEFAULT_SPLIT_SEED,
    assign_splits_by_source_image,
    split_counts,
)
from src.dataset.leakage import METADATA_FIELDS
from src.dataset.taxonomy import CLASS_TO_ID, DATASET_VERSION, TAXONOMY_VERSION
from src.utils.paths import project_root

VALID_SPLITS = ("train", "val", "test")

PROVENANCE_EXPORT_FIELDS = (
    "sample_id",
    "source_dataset",
    "source_image",
    "original_class",
    "source_subtype",
    "mapped_class",
    "direction_status",
    "direction_class",
    "taxonomy_version",
    "license",
    "original_annotation",
    "split",
    "filename",
)


class PromotionError(ValueError):
    """Raised when promotion would leak sequences or copy unreviewed data."""


def promote_staging(
    staging_metadata: Path,
    dataset_root: Path | None = None,
    *,
    dry_run: bool = False,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, int]:
    """Copy approved staging samples into data/li_stick_navigation."""
    dataset_root = dataset_root or (project_root() / "data" / "li_stick_navigation")
    records = read_provenance(staging_metadata)
    assigned = assign_splits_by_source_image(
        [record for record in records if class_for_promotion(record)],
        seed=split_seed,
    )
    if not assigned:
        return {"promoted_boxes": 0, "images": 0}

    planned = _plan_by_image(assigned)
    _assert_sequence_split_safety(planned)
    _assert_source_image_split_safety(planned)

    copied_images = 0
    copied_boxes = 0
    exported: list[ProvenanceRecord] = []
    for (staging_image, split), group in planned.items():
        if split not in VALID_SPLITS:
            raise PromotionError(
                f"Sample {group[0].sample_id} has no train/val/test split "
                f"({split!r}). Assign a source split before promotion."
            )
        dest_image = dataset_root / "images" / split / Path(staging_image).name
        dest_label = dataset_root / "labels" / split / f"{Path(staging_image).stem}.txt"
        lines = [_yolo_line(record) for record in group]
        if dry_run:
            copied_images += 1
            copied_boxes += len(lines)
            exported.extend(group)
            continue
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        source_image = Path(staging_image)
        if not source_image.is_file():
            raise PromotionError(f"Missing staging image: {source_image}")
        shutil.copy2(source_image, dest_image)
        dest_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        copied_images += 1
        copied_boxes += len(lines)
        exported.extend(group)

    counts = split_counts(exported)
    summary = {
        "promoted_boxes": copied_boxes,
        "images": copied_images,
        "train_images": counts["train"]["images"],
        "val_images": counts["val"]["images"],
        "test_images": counts["test"]["images"],
        "train_annotations": counts["train"]["annotations"],
        "val_annotations": counts["val"]["annotations"],
        "test_annotations": counts["test"]["annotations"],
        "split_seed": split_seed,
    }
    if not dry_run:
        _write_dataset_sidecar(dataset_root, exported)
        _sync_detector_splits(staging_metadata, assigned, records)
    return summary


def _plan_by_image(
    records: list[ProvenanceRecord],
) -> dict[tuple[str, str], list[ProvenanceRecord]]:
    grouped: dict[tuple[str, str], list[ProvenanceRecord]] = defaultdict(list)
    for record in records:
        split = (record.detector_split or record.source_split or "").lower()
        if split in {"valid", "validation"}:
            split = "val"
        grouped[(record.staging_image, split)].append(record)
    return grouped


def _assert_sequence_split_safety(
    planned: dict[tuple[str, str], list[ProvenanceRecord]],
) -> None:
    by_sequence: dict[str, set[str]] = defaultdict(set)
    for (_image, split), group in planned.items():
        for record in group:
            if record.sequence_id:
                by_sequence[record.sequence_id].add(split)
    leaks = {
        sequence: splits
        for sequence, splits in by_sequence.items()
        if len(splits) > 1
    }
    if leaks:
        details = ", ".join(
            f"{seq} -> {sorted(splits)}" for seq, splits in sorted(leaks.items())
        )
        raise PromotionError(
            "Refusing to promote: the same sequence_id would appear in multiple "
            f"splits ({details}). Keep each video/session/location in one split."
        )


def _assert_source_image_split_safety(
    planned: dict[tuple[str, str], list[ProvenanceRecord]],
) -> None:
    by_source: dict[str, set[str]] = defaultdict(set)
    for (_image, split), group in planned.items():
        for record in group:
            if record.source_image:
                by_source[record.source_image].add(split)
    leaks = {
        image: splits
        for image, splits in by_source.items()
        if len(splits) > 1
    }
    if leaks:
        details = ", ".join(
            f"{image} -> {sorted(splits)}" for image, splits in sorted(leaks.items())[:5]
        )
        raise PromotionError(
            "Refusing to promote: the same source_image would appear in multiple "
            f"splits ({details})."
        )


def _yolo_line(record: ProvenanceRecord) -> str:
    mapped = class_for_promotion(record)
    if mapped is None or mapped not in CLASS_TO_ID:
        raise PromotionError(f"{record.sample_id} has no promotable custom class.")
    box, width, height = _box_from_annotation(record.original_annotation)
    yolo = xyxy_to_yolo(box, width, height, CLASS_TO_ID[mapped])
    return (
        f"{yolo.class_id} {yolo.x_center:.6f} {yolo.y_center:.6f} "
        f"{yolo.width:.6f} {yolo.height:.6f}"
    )


def _box_from_annotation(text: str) -> tuple[PixelBox, int, int]:
    if "xyxy=" not in text or "wh=" not in text:
        raise PromotionError(
            f"Cannot rebuild a YOLO box from original_annotation={text!r}."
        )
    xyxy_part = text.split("xyxy=", 1)[1].split(";", 1)[0]
    x1, y1, x2, y2 = (float(part) for part in xyxy_part.split(","))
    size_part = text.split("wh=", 1)[1].split(";", 1)[0]
    width_s, height_s = size_part.lower().split("x")
    return PixelBox(x1, y1, x2, y2), int(float(width_s)), int(float(height_s))


def _sync_detector_splits(
    staging_metadata: Path,
    assigned: list[ProvenanceRecord],
    all_records: list[ProvenanceRecord],
) -> None:
    """Persist rebuilt detector_split values back into staging metadata."""
    split_by_id = {
        record.sample_id: record.detector_split
        for record in assigned
        if record.detector_split
    }
    if not split_by_id:
        return
    merged = [
        replace(record, detector_split=split_by_id.get(record.sample_id, record.detector_split))
        if record.sample_id in split_by_id
        else record
        for record in all_records
    ]
    write_provenance(staging_metadata, merged)


def _write_dataset_sidecar(dataset_root: Path, records: list[ProvenanceRecord]) -> None:
    metadata_path = dataset_root / "metadata.csv"
    provenance_path = dataset_root / "provenance.csv"
    by_filename: dict[str, ProvenanceRecord] = {}
    for record in records:
        filename = Path(record.staging_image).name
        by_filename[filename] = record
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for filename, record in sorted(by_filename.items()):
            split = (record.detector_split or record.source_split or "").lower()
            writer.writerow(
                {
                    "filename": filename,
                    "split": split,
                    "video_id": "",
                    "session_id": "",
                    "location_id": "",
                    "sequence_id": record.source_image,
                    "frame_index": "",
                    "notes": (
                        f"taxonomy={TAXONOMY_VERSION}; dataset={DATASET_VERSION}; "
                        f"source={record.source_dataset}; license={record.license}"
                    ),
                }
            )
    with provenance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROVENANCE_EXPORT_FIELDS)
        writer.writeheader()
        for record in records:
            split = (record.detector_split or record.source_split or "").lower()
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "source_dataset": record.source_dataset,
                    "source_image": record.source_image,
                    "original_class": record.original_class,
                    "source_subtype": record.source_subtype,
                    "mapped_class": class_for_promotion(record) or record.mapped_class,
                    "direction_status": record.direction_status,
                    "direction_class": record.direction_class,
                    "taxonomy_version": record.taxonomy_version or TAXONOMY_VERSION,
                    "license": record.license,
                    "original_annotation": record.original_annotation,
                    "split": split,
                    "filename": Path(record.staging_image).name,
                }
            )
