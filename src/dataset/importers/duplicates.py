"""Exact-duplicate detection for staging review queues.

Near-duplicates are reported only when the same source path or file
bytes match. Perceptual similarity is not used and nothing is discarded.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.dataset.importers.provenance import ProvenanceRecord


@dataclass(frozen=True)
class DuplicateGroup:
    kind: str
    key: str
    sample_ids: tuple[str, ...]
    source_images: tuple[str, ...]


def find_exact_image_duplicates(
    records: list[ProvenanceRecord],
    *,
    hash_bytes: bool = True,
) -> list[DuplicateGroup]:
    """Find exact duplicate *images*, not multiple boxes on one image."""
    by_source_path: dict[str, list[ProvenanceRecord]] = defaultdict(list)
    unique_files: dict[str, ProvenanceRecord] = {}
    for record in records:
        source = record.source_image or record.staging_image
        by_source_path[source].append(record)
        if source not in unique_files:
            unique_files[source] = record

    groups: list[DuplicateGroup] = []
    if hash_bytes:
        digest_to_sources: dict[str, list[str]] = defaultdict(list)
        for source, record in unique_files.items():
            path = Path(record.staging_image or source)
            if not path.is_file():
                continue
            digest_to_sources[_md5_file(path)].append(source)
        for digest, sources in digest_to_sources.items():
            if len(sources) < 2:
                continue
            sample_ids = tuple(
                record.sample_id
                for source in sources
                for record in by_source_path[source]
            )
            groups.append(
                DuplicateGroup(
                    kind="exact_bytes",
                    key=digest,
                    sample_ids=sample_ids,
                    source_images=tuple(sources),
                )
            )
    return groups


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
