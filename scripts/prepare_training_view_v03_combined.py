"""Build the v0.3-combined derived view. Does not modify earlier experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.class_map import v03_combined_class_map, write_class_map
from src.dataset.combined_view import (
    CANONICAL_DATASET,
    COCO_RAW,
    COMBINED_VIEW,
    ROBOFLOW_RAW,
    assemble_combined_boxes,
    boxes_to_split_rows,
    write_combined_view,
)
from src.dataset.leakage import find_split_leakage, parse_metadata_csv
from src.dataset.split_report import count_split_classes, format_balance_report
from src.dataset.stratified_split import (
    COMBINED_V03_CLASSES,
    COMBINED_V03_MIN_TEST,
    COMBINED_V03_MIN_VAL,
    DEFAULT_SPLIT_SEED,
    assign_group_aware_splits,
    build_groups,
    leakage_report,
    write_split_assignment,
)
from src.dataset.validator import validate_dataset
from src.utils.paths import project_root, resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=CANONICAL_DATASET)
    parser.add_argument("--roboflow", type=Path, default=ROBOFLOW_RAW)
    parser.add_argument("--coco", type=Path, default=COCO_RAW)
    parser.add_argument("--dest", type=Path, default=COMBINED_VIEW)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-coco",
        action="store_true",
        help="Download/filter the official COCO chair subset before building.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source if args.source.is_absolute() else resolve_from_root(args.source)
    roboflow = args.roboflow if args.roboflow.is_absolute() else resolve_from_root(args.roboflow)
    coco = args.coco if args.coco.is_absolute() else resolve_from_root(args.coco)
    dest = args.dest if args.dest.is_absolute() else resolve_from_root(args.dest)

    if args.prepare_coco:
        from src.dataset.coco_generic_subset import prepare_coco_chair_subset

        coco_stats = prepare_coco_chair_subset(coco, download=True)
        print(
            f"COCO subset: images={coco_stats.images} "
            f"annotations={coco_stats.annotations} per_class={coco_stats.per_class}"
        )

    class_map = v03_combined_class_map()
    boxes, stats = assemble_combined_boxes(
        canonical_root=source,
        roboflow_root=roboflow,
        coco_root=coco,
        class_map=class_map,
    )
    rows = boxes_to_split_rows(boxes)
    groups = build_groups(rows, class_names=COMBINED_V03_CLASSES)
    result = assign_group_aware_splits(
        groups,
        seed=args.seed,
        min_val=COMBINED_V03_MIN_VAL,
        min_test=COMBINED_V03_MIN_TEST,
        class_names=COMBINED_V03_CLASSES,
        protect_train_share=0.6,
    )
    leaks = leakage_report(result)
    stats["split_images"] = result.image_counts
    stats["split_annotations"] = result.box_counts
    stats["per_class_by_split"] = result.class_counts
    stats["leakage"] = leaks
    stats["split_notes"] = result.notes

    print("=== v0.3-combined pre-train report ===")
    print("TOTAL images:", stats["images"])
    print("TOTAL annotations:", stats["annotations"])
    print("Source contribution:", stats["source_contribution"])
    print("Custom/generic ratio:", stats["custom_generic_ratio"])
    print("Per class (instances):")
    for name in COMBINED_V03_CLASSES:
        split_counts = result.class_counts
        print(
            f"  {name}: total={stats['per_class'][name]} "
            f"train={split_counts['train'][name]} "
            f"val={split_counts['val'][name]} "
            f"test={split_counts['test'][name]}"
        )
    print("Split images:", result.image_counts)
    print(
        "Leakage: "
        f"source_image={leaks['source_image_leakage']} "
        f"group={leaks['group_leakage']} "
        f"exact_duplicate={leaks['exact_duplicate_leakage']}"
    )
    for note in result.notes:
        print("Note:", note)

    ratio = stats["custom_generic_ratio"]
    if ratio.get("severe_imbalance"):
        print("FLAG: generic instances overwhelm custom by >=20x.")

    zero = [
        f"{split}:{name}"
        for split in ("val", "test")
        for name in COMBINED_V03_CLASSES
        if result.class_counts[split][name] <= 0
    ]
    if leaks["source_image_leakage"] != 0 or leaks["group_leakage"] != 0:
        print("STOP: leakage is not 0.")
        return 1
    if zero:
        print("STOP: class missing from val/test:", ", ".join(zero))
        return 1
    if stats["per_class"].get("chair", 0) <= 0:
        print("STOP: chair has zero instances. Prepare the official COCO subset.")
        return 1
    if args.dry_run:
        (dest / "pretrain_report.dry_run.json").parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(stats, indent=2)[:4000])
        print("Dry-run: derived view not written.")
        return 0

    write_combined_view(
        dest,
        boxes,
        result.filename_to_split,
        class_map,
        partial_label=stats["partial_label"],
        stats=stats,
    )
    write_split_assignment(
        dest / "split_assignment.csv",
        rows,
        result,
        split_column="v03_detector_split",
    )
    sidecar = project_root() / "models" / "li_stick_navigation_v0.3_combined.class_map.yaml"
    write_class_map(sidecar, class_map)

    counts = count_split_classes(dest, class_map=class_map)
    print(f"Training view written to {dest}")
    print(f"Class map written to {sidecar}")
    print(format_balance_report(counts, classes=COMBINED_V03_CLASSES))

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


if __name__ == "__main__":
    raise SystemExit(main())
