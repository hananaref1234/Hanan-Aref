"""Validate staging_metadata.csv. Does not train or promote."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers.provenance import default_metadata_path, read_provenance
from src.dataset.importers.staging_validate import validate_staging_records
from src.dataset.validator import validate_dataset
from src.utils.paths import project_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        default=str(project_root() / "data" / "staging"),
    )
    args = parser.parse_args()
    metadata = default_metadata_path(Path(args.staging_root))
    records = read_provenance(metadata)
    issues = validate_staging_records(records)
    print(f"Staging rows: {len(records)}")
    if issues:
        print(f"STAGING ISSUES: {len(issues)}")
        for item in issues[:40]:
            print(" ", item)
        return 1
    print("STAGING: OK")
    report = validate_dataset(project_root() / "data" / "li_stick_navigation")
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
