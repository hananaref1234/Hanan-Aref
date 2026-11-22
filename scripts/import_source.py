"""Convert a raw source into staging. Does not train. Does not download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers import importer_for
from src.dataset.importers.provenance import (
    default_metadata_path,
    read_provenance,
    write_provenance,
)
from src.dataset.license_gate import LicenseGateError, assert_import_allowed
from src.dataset.manifest import get_source
from src.utils.paths import project_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Manifest source name.")
    parser.add_argument(
        "--include-generic",
        action="store_true",
        help="Also stage combined-taxonomy classes (person, car, ...).",
    )
    parser.add_argument(
        "--replace-source",
        action="store_true",
        help="Replace existing staging rows for this source instead of appending.",
    )
    parser.add_argument(
        "--staging-root",
        default=str(project_root() / "data" / "staging"),
    )
    args = parser.parse_args()

    source = get_source(args.source)
    if source.status == "REJECTED":
        print(f"Refusing to import rejected source {source.name}.")
        return 2
    try:
        assert_import_allowed(source)
    except LicenseGateError as exc:
        print(exc)
        return 2
    staging_root = Path(args.staging_root)
    importer = importer_for(source, staging_root=staging_root)
    result = importer.convert(include_generic=args.include_generic)
    metadata_path = default_metadata_path(staging_root)
    existing = read_provenance(metadata_path)
    if args.replace_source:
        existing = [row for row in existing if row.source_dataset != source.name]
    write_provenance(metadata_path, existing + result.records)
    print(
        f"Staged {len(result.records)} objects from {source.name} "
        f"(skipped={result.skipped}, errors={len(result.errors)})."
    )
    print(f"Provenance: {metadata_path}")
    if result.errors:
        for message in result.errors[:20]:
            print(f"  error: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
