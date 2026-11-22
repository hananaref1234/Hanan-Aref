"""Inspect configured dataset sources without downloading anything."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers import importer_for
from src.dataset.importers.download import acquisition_instructions
from src.dataset.manifest import get_source, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Inspect one source name from the manifest.")
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print human download/access instructions.",
    )
    args = parser.parse_args()

    sources = [get_source(args.source)] if args.source else load_manifest()
    print(f"{'NAME':<24} {'STATUS':<28} {'PRESENT':<8} IMAGES  ANNOT  LICENSE")
    for source in sources:
        importer = importer_for(source)
        report = importer.inspect()
        present = "yes" if report.present else "no"
        print(
            f"{source.name:<24} {source.status:<28} {present:<8} "
            f"{report.image_files:<6} {report.annotation_files:<6} {source.license}"
        )
        if args.instructions:
            print()
            print(acquisition_instructions(source))
            print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
