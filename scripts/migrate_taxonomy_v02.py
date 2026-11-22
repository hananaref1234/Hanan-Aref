"""Rebuild Accessibility Barriers staging for taxonomy v0.2 and promote it.

Preserves v0.1 STAIRS_*/CURB_* / REJECTED human decisions as direction metadata.
Does not train.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers import importer_for
from src.dataset.importers.boxes import xyxy_to_yolo
from src.dataset.importers.coverage import coverage_from_records, format_coverage_table
from src.dataset.importers.promote import PromotionError, promote_staging
from src.dataset.importers.provenance import (
    default_metadata_path,
    read_provenance,
    write_provenance,
)
from src.dataset.importers.review import class_for_promotion
from src.dataset.importers.source_inventory import inventory_accessibility_barriers
from src.dataset.importers.splits import DEFAULT_SPLIT_SEED
from src.dataset.importers.staging_validate import validate_staging_records
from src.dataset.importers.v01_preserve import preserve_v01_direction_metadata
from src.dataset.manifest import get_source
from src.dataset.taxonomy import CLASS_TO_ID, TAXONOMY_VERSION
from src.dataset.validator import validate_dataset
from src.utils.paths import project_root

V01_ARCHIVE_NAME = "staging_metadata_v0.1_directional.csv"


def rewrite_staging_labels(records) -> None:
    grouped: dict[str, list] = defaultdict(list)
    for record in records:
        if not record.staging_label or record.staging_label in {".", ""}:
            continue
        grouped[record.staging_label].append(record)
    for label_path, group in grouped.items():
        path = Path(label_path)
        lines: list[str] = []
        for record in group:
            mapped = class_for_promotion(record)
            if mapped not in CLASS_TO_ID:
                continue
            if "xyxy=" not in record.original_annotation or "wh=" not in record.original_annotation:
                continue
            xyxy_part = record.original_annotation.split("xyxy=", 1)[1].split(";", 1)[0]
            x1, y1, x2, y2 = (float(part) for part in xyxy_part.split(","))
            size_part = record.original_annotation.split("wh=", 1)[1].split(";", 1)[0]
            width_s, height_s = size_part.lower().split("x")
            from src.dataset.importers.boxes import PixelBox

            yolo = xyxy_to_yolo(
                PixelBox(x1, y1, x2, y2),
                int(float(width_s)),
                int(float(height_s)),
                CLASS_TO_ID[mapped],
            )
            lines.append(
                f"{yolo.class_id} {yolo.x_center:.6f} {yolo.y_center:.6f} "
                f"{yolo.width:.6f} {yolo.height:.6f}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")


def migrate(staging_root: Path, dataset_root: Path, *, dry_run: bool) -> dict:
    source = get_source("accessibility_barriers")
    metadata_path = default_metadata_path(staging_root)
    archive_path = staging_root / V01_ARCHIVE_NAME
    old_records = read_provenance(metadata_path)
    if old_records and not archive_path.is_file():
        shutil.copy2(metadata_path, archive_path)
    archive_records = (
        read_provenance(archive_path) if archive_path.is_file() else old_records
    )

    importer = importer_for(source, staging_root=staging_root)
    result = importer.convert(include_generic=False)
    merged = preserve_v01_direction_metadata(result.records, archive_records)
    current = read_provenance(metadata_path)
    others = [row for row in current if row.source_dataset != source.name]
    if not dry_run:
        write_provenance(metadata_path, others + merged)
        rewrite_staging_labels(merged)

    inventory = inventory_accessibility_barriers(source.raw_path)
    staging_issues = validate_staging_records(merged)
    promoted = {"promoted_boxes": 0, "images": 0}
    dataset_report = None
    if not dry_run:
        try:
            promoted = promote_staging(metadata_path, dataset_root, split_seed=DEFAULT_SPLIT_SEED)
        except PromotionError as exc:
            promoted = {"error": str(exc)}
        dataset_report = validate_dataset(dataset_root)

    coverage_rows = coverage_from_records(merged)
    stairs = [row for row in merged if class_for_promotion(row) == "stairs"]
    curbs = [row for row in merged if class_for_promotion(row) == "curb"]
    preserved = sum(1 for row in merged if row.direction_status in {
        "STAIRS_UP", "STAIRS_DOWN", "CURB_UP", "CURB_DOWN", "REJECTED", "AMBIGUOUS"
    })
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "xml_stairs": inventory.labels.get("stair", 0) + inventory.labels.get("stairs", 0),
        "xml_curb": inventory.step_types.get("curb", 0),
        "xml_annotations": inventory.annotations,
        "staged_records": len(merged),
        "import_errors": result.errors,
        "preserved_direction_rows": preserved,
        "promotable_stairs_boxes": len(stairs),
        "promotable_stairs_images": len({row.source_image for row in stairs}),
        "promotable_curb_boxes": len(curbs),
        "promotable_curb_images": len({row.source_image for row in curbs}),
        "staging_issues": staging_issues,
        "promoted": promoted,
        "dataset_passed": None if dataset_report is None else dataset_report.passed,
        "dataset_summary": None if dataset_report is None else dataset_report.summary(),
        "coverage_table": format_coverage_table(coverage_rows),
        "archive_path": str(archive_path),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--staging-root",
        default=str(project_root() / "data" / "staging"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(project_root() / "data" / "li_stick_navigation"),
    )
    args = parser.parse_args()
    payload = migrate(Path(args.staging_root), Path(args.dataset_root), dry_run=args.dry_run)
    print(f"taxonomy_version: {payload['taxonomy_version']}")
    print(f"XML stairs={payload['xml_stairs']} curb={payload['xml_curb']}")
    print(
        "Promotable stairs "
        f"{payload['promotable_stairs_images']} images / "
        f"{payload['promotable_stairs_boxes']} boxes"
    )
    print(
        "Promotable curb "
        f"{payload['promotable_curb_images']} images / "
        f"{payload['promotable_curb_boxes']} boxes"
    )
    print(f"Preserved v0.1 direction/reject rows: {payload['preserved_direction_rows']}")
    print(f"Archive: {payload['archive_path']}")
    print(payload["coverage_table"])
    if payload["staging_issues"]:
        print("Staging issues:")
        for issue in payload["staging_issues"][:20]:
            print(f"  {issue}")
        return 1
    print(f"Promoted: {payload['promoted']}")
    if payload["dataset_summary"]:
        print(payload["dataset_summary"])
    print("TRAINING STATUS: NOT STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
