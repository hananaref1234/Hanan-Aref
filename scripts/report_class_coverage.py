"""Print custom-class coverage from staging provenance. Does not fabricate counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers.coverage import (
    coverage_from_staging,
    format_coverage_markdown,
    format_coverage_table,
)
from src.dataset.importers.provenance import default_metadata_path
from src.utils.paths import project_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        default=str(project_root() / "data" / "staging"),
    )
    parser.add_argument(
        "--write",
        default="",
        help="Optional markdown path to write, e.g. data/sources/CLASS_COVERAGE.md",
    )
    args = parser.parse_args()

    metadata = default_metadata_path(Path(args.staging_root))
    rows = coverage_from_staging(metadata)
    table = format_coverage_table(rows)
    print(table)
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_coverage_markdown(rows), encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
