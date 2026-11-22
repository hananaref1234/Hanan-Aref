"""Print (or optionally run) legitimate source acquisition steps.

Automatic download is disabled for access-controlled and license-review sources.
The only opt-in public download is accessibility_barriers, and only with --download.
This script does not train a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.importers.download import (
    DownloadBlockedError,
    acquisition_instructions,
    assert_download_allowed,
)
from src.dataset.manifest import get_source, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="One source name. Default: print all.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Opt-in download for the few public sources that allow it.",
    )
    args = parser.parse_args()

    sources = [get_source(args.source)] if args.source else load_manifest()
    for source in sources:
        print(acquisition_instructions(source))
        print()
        if args.download:
            try:
                assert_download_allowed(source)
            except DownloadBlockedError as exc:
                print(exc)
                return 2
            dest = source.raw_path / Path(source.download_url).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {source.download_url} -> {dest}")
            urlretrieve(source.download_url, dest)
            print("Download finished. Extract the archive into the raw directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
