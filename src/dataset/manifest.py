"""Load and validate data/sources/datasets.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import project_root

ALLOWED_STATUSES = frozenset(
    {
        "APPROVED",
        "MANUAL_DOWNLOAD_REQUIRED",
        "MANUAL_REVIEW_REQUIRED",
        "LICENSE_REVIEW_REQUIRED",
        "REJECTED",
    }
)

ALLOWED_MAPPING_TYPES = frozenset(
    {
        "DIRECT",
        "MANUAL_DIRECTION",
        "SEMANTIC_REVIEW",
        "MASK_TO_BOX",
        "CUSTOM",
    }
)

REQUIRED_SOURCE_FIELDS = (
    "name",
    "official_url",
    "license",
    "license_url",
    "date_accessed",
    "download_method",
    "annotation_format",
    "source_classes",
    "listick_mapping",
    "status",
    "notes",
)


class ManifestError(ValueError):
    """Raised when the source manifest is missing or invalid."""


@dataclass(frozen=True)
class MappingSpec:
    target: str | None
    mapping_type: str
    auto: bool = False
    review: str | None = None
    taxonomy: str = "custom"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    official_url: str
    license: str
    license_url: str
    date_accessed: str
    download_method: str
    annotation_format: str
    source_classes: tuple[str, ...]
    listick_mapping: dict[str, MappingSpec]
    status: str
    notes: str
    raw_dir: str
    download_url: str = ""
    request_url: str = ""
    disk_space_hint: str = ""
    extra: dict[str, Any] | None = None

    @property
    def raw_path(self) -> Path:
        return project_root() / self.raw_dir


def default_manifest_path() -> Path:
    return project_root() / "data" / "sources" / "datasets.yaml"


def load_manifest(path: Path | None = None) -> list[SourceSpec]:
    """Parse the source manifest and validate required fields."""
    manifest_path = path or default_manifest_path()
    if not manifest_path.is_file():
        raise ManifestError(f"Source manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "sources" not in data:
        raise ManifestError(f"{manifest_path} must contain a top-level 'sources' list.")
    sources = data["sources"]
    if not isinstance(sources, list) or not sources:
        raise ManifestError(f"{manifest_path} 'sources' must be a non-empty list.")
    parsed = [_parse_source(item, manifest_path) for item in sources]
    names = [source.name for source in parsed]
    if len(names) != len(set(names)):
        raise ManifestError("Duplicate source names in the manifest.")
    return parsed


def get_source(name: str, path: Path | None = None) -> SourceSpec:
    """Return one source by name."""
    for source in load_manifest(path):
        if source.name == name:
            return source
    raise ManifestError(f"Unknown source {name!r}.")


def _parse_source(item: Any, manifest_path: Path) -> SourceSpec:
    if not isinstance(item, dict):
        raise ManifestError(f"{manifest_path}: each source must be a mapping.")
    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in item]
    if missing:
        raise ManifestError(
            f"{manifest_path}: source {item.get('name', '<unnamed>')} missing "
            f"fields: {', '.join(missing)}."
        )
    status = str(item["status"]).strip()
    if status not in ALLOWED_STATUSES:
        raise ManifestError(
            f"Invalid status {status!r} for source {item['name']!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_STATUSES))}."
        )
    source_classes = item["source_classes"]
    if source_classes is None:
        source_classes = []
    if not isinstance(source_classes, list):
        raise ManifestError(f"source_classes for {item['name']!r} must be a list.")
    mapping_raw = item["listick_mapping"] or {}
    if not isinstance(mapping_raw, dict):
        raise ManifestError(f"listick_mapping for {item['name']!r} must be a mapping.")
    mapping = {
        str(key): _parse_mapping(item["name"], str(key), value)
        for key, value in mapping_raw.items()
    }
    known_extra = {
        "name",
        "official_url",
        "license",
        "license_url",
        "date_accessed",
        "download_method",
        "annotation_format",
        "source_classes",
        "listick_mapping",
        "status",
        "notes",
        "raw_dir",
        "download_url",
        "request_url",
        "disk_space_hint",
        "selection_required",
    }
    extra = {key: value for key, value in item.items() if key not in known_extra}
    return SourceSpec(
        name=str(item["name"]).strip(),
        official_url=str(item["official_url"]),
        license=str(item["license"]),
        license_url=str(item["license_url"]),
        date_accessed=str(item["date_accessed"]),
        download_method=str(item["download_method"]),
        annotation_format=str(item["annotation_format"]),
        source_classes=tuple(str(name) for name in source_classes),
        listick_mapping=mapping,
        status=status,
        notes=str(item["notes"]).strip(),
        raw_dir=str(item.get("raw_dir") or f"data/raw/{item['name']}"),
        download_url=str(item.get("download_url") or ""),
        request_url=str(item.get("request_url") or ""),
        disk_space_hint=str(item.get("disk_space_hint") or ""),
        extra=extra or None,
    )


def _parse_mapping(source_name: str, original: str, value: Any) -> MappingSpec:
    if not isinstance(value, dict):
        raise ManifestError(
            f"Mapping for {source_name!r} class {original!r} must be a mapping."
        )
    mapping_type = str(value.get("mapping_type", "CUSTOM")).strip()
    if mapping_type not in ALLOWED_MAPPING_TYPES:
        raise ManifestError(
            f"Invalid mapping_type {mapping_type!r} for {source_name}:{original}."
        )
    target = value.get("target")
    if target is not None:
        target = str(target).strip() or None
    return MappingSpec(
        target=target,
        mapping_type=mapping_type,
        auto=bool(value.get("auto", False)),
        review=str(value["review"]).strip() if value.get("review") else None,
        taxonomy=str(value.get("taxonomy", "custom")).strip() or "custom",
    )
