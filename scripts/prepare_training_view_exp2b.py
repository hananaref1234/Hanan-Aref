"""Build the Exp2b derived view with group-aware stratified splits.

Does not modify canonical data, Exp1, or Exp2.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.class_map import exp2b_six_class_class_map, write_class_map
from src.dataset.importers.promote import PROVENANCE_EXPORT_FIELDS
from src.dataset.leakage import METADATA_FIELDS, find_split_leakage, parse_metadata_csv
from src.dataset.split_report import count_split_classes, format_balance_report
from src.dataset.stratified_split import (
    DEFAULT_SPLIT_SEED,
    EXP2B_CLASSES,
    assign_group_aware_splits,
    build_groups,
    leakage_report,
    rows_from_canonical_provenance,
    write_split_assignment,
)
from src.dataset.training_view import (
    CANONICAL_DATASET,
    EXP2B_TRAINING_VIEW,
    build_training_view,
)
from src.dataset.validator import validate_dataset
from src.utils.paths import project_root, resolve_from_root


EXP2B_PROVENANCE_FIELDS = PROVENANCE_EXPORT_FIELDS + (
    "original_source_split",
    "exp2b_detector_split",
    "group_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=CANONICAL_DATASET)
    parser.add_argument("--dest", type=Path, default=EXP2B_TRAINING_VIEW)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source if args.source.is_absolute() else resolve_from_root(args.source)
    dest = args.dest if args.dest.is_absolute() else resolve_from_root(args.dest)
    provenance_path = source / "provenance.csv"
    rows = rows_from_canonical_provenance(provenance_path)
    groups = build_groups(rows)
    result = assign_group_aware_splits(groups, seed=args.seed)
    leaks = leakage_report(result)

    print("Approved boxes used:", len(rows))
    print("Unique images:", len(result.filename_to_split))
    print("Groups:", len(groups))
    print("Split images:", result.image_counts)
    print("Split annotations:", result.box_counts)
    print("Per-class instances:")
    for name in EXP2B_CLASSES:
        print(
            f"  {name}: train={result.class_counts['train'][name]} "
            f"val={result.class_counts['val'][name]} "
            f"test={result.class_counts['test'][name]}"
        )
    print(
        "Leakage: "
        f"source_overlap={leaks['source_overlap']} "
        f"source-image={leaks['source_image_leakage']} "
        f"exact_duplicate={leaks['exact_duplicate_leakage']}"
    )
    for note in result.notes:
        print("Note:", note)

    zero = [
        f"{split}:{name}"
        for split in ("val", "test")
        for name in EXP2B_CLASSES
        if result.class_counts[split][name] <= 0
    ]
    if leaks["source_image_leakage"] != 0:
        print("STOP: source-image leakage is not 0.")
        return 1
    if zero:
        print("STOP: class missing from val/test:", ", ".join(zero))
        return 1
    if args.dry_run:
        print("Dry-run: derived view not written.")
        return 0

    class_map = exp2b_six_class_class_map()
    stats = build_training_view(
        source,
        dest,
        class_map,
        split_by_filename=result.filename_to_split,
        copy_sidecars=False,
    )
    write_split_assignment(dest / "split_assignment.csv", rows, result)
    _write_exp2b_sidecars(source / "provenance.csv", dest, rows, result)
    sidecar = project_root() / "models" / "li_stick_navigation_v0.2_exp2b.class_map.yaml"
    write_class_map(sidecar, class_map)

    counts = count_split_classes(dest, class_map=class_map)
    print(f"Training view written to {stats.dest_root}")
    print(f"Class map written to {sidecar}")
    print(format_balance_report(counts, classes=EXP2B_CLASSES))

    report = validate_dataset(dest)
    print(report.summary())
    if not report.passed:
        print("STOP: dataset validation failed.")
        return 1
    meta_leaks = find_split_leakage(parse_metadata_csv(dest / "metadata.csv"))
    print("Metadata group leakage issues:", len(meta_leaks))
    if meta_leaks:
        print("STOP: metadata leakage detected.")
        return 1
    return 0


def _write_exp2b_sidecars(
    canonical_provenance: Path,
    dest: Path,
    rows,
    result,
) -> None:
    by_sample = {row.sample_id: row for row in rows}
    with canonical_provenance.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        exported = []
        for raw in reader:
            sample_id = str(raw.get("sample_id", "")).strip()
            plan = by_sample.get(sample_id)
            if plan is None:
                continue
            split = result.filename_to_split[plan.filename]
            exported.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": raw.get("source_dataset", ""),
                    "source_image": raw.get("source_image", ""),
                    "original_class": raw.get("original_class", ""),
                    "source_subtype": raw.get("source_subtype", ""),
                    "mapped_class": raw.get("mapped_class", ""),
                    "direction_status": raw.get("direction_status", ""),
                    "direction_class": raw.get("direction_class", ""),
                    "taxonomy_version": raw.get("taxonomy_version", ""),
                    "license": raw.get("license", ""),
                    "original_annotation": raw.get("original_annotation", ""),
                    "split": split,
                    "filename": plan.filename,
                    "original_source_split": plan.original_source_split,
                    "exp2b_detector_split": split,
                    "group_id": plan.group_id,
                }
            )
    with (dest / "provenance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXP2B_PROVENANCE_FIELDS)
        writer.writeheader()
        writer.writerows(exported)

    by_filename = {}
    for item in exported:
        by_filename[item["filename"]] = item
    with (dest / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for filename, item in sorted(by_filename.items()):
            writer.writerow(
                {
                    "filename": filename,
                    "split": item["exp2b_detector_split"],
                    "video_id": "",
                    "session_id": "",
                    "location_id": "",
                    "sequence_id": item["group_id"],
                    "frame_index": "",
                    "notes": (
                        f"exp2b_split={item['exp2b_detector_split']}; "
                        f"original_source_split={item['original_source_split']}; "
                        f"source={item['source_dataset']}"
                    ),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
