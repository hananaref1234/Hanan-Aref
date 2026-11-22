"""Train/val/test leakage checks for video-derived frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

METADATA_FIELDS = (
    "filename",
    "split",
    "video_id",
    "session_id",
    "location_id",
    "sequence_id",
    "frame_index",
    "notes",
)

GROUP_FIELDS = ("video_id", "session_id", "location_id", "sequence_id")


@dataclass(frozen=True)
class MetadataRow:
    filename: str
    split: str
    video_id: str = ""
    session_id: str = ""
    location_id: str = ""
    sequence_id: str = ""
    frame_index: str = ""
    notes: str = ""

    def group_keys(self) -> dict[str, str]:
        return {
            "video_id": self.video_id.strip(),
            "session_id": self.session_id.strip(),
            "location_id": self.location_id.strip(),
            "sequence_id": self.sequence_id.strip(),
        }


@dataclass(frozen=True)
class LeakageIssue:
    group_field: str
    group_value: str
    splits: tuple[str, ...]
    filenames: tuple[str, ...]


def parse_metadata_csv(path: Path) -> list[MetadataRow]:
    """Parse a simple comma-separated metadata table.

    The first row must be a header containing at least ``filename`` and ``split``.
    """
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        required = {"filename", "split"}
        missing = required - {name.strip() for name in reader.fieldnames}
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}."
            )
        rows: list[MetadataRow] = []
        for raw in reader:
            rows.append(
                MetadataRow(
                    filename=str(raw.get("filename", "")).strip(),
                    split=str(raw.get("split", "")).strip().lower(),
                    video_id=str(raw.get("video_id", "")).strip(),
                    session_id=str(raw.get("session_id", "")).strip(),
                    location_id=str(raw.get("location_id", "")).strip(),
                    sequence_id=str(raw.get("sequence_id", "")).strip(),
                    frame_index=str(raw.get("frame_index", "")).strip(),
                    notes=str(raw.get("notes", "")).strip(),
                )
            )
        return rows


def find_split_leakage(rows: list[MetadataRow]) -> list[LeakageIssue]:
    """Find groups that appear in more than one of train/val/test.

    Consecutive walking-video frames are nearly identical. If the same
    video/session/location/sequence is present in two splits, the test set
    is contaminated.
    """
    issues: list[LeakageIssue] = []
    for field in GROUP_FIELDS:
        grouped: dict[str, dict[str, list[str]]] = {}
        for row in rows:
            value = row.group_keys()[field]
            if not value:
                continue
            by_split = grouped.setdefault(value, {})
            by_split.setdefault(row.split, []).append(row.filename)
        for value, by_split in grouped.items():
            splits = tuple(sorted(by_split))
            if len(splits) < 2:
                continue
            filenames: list[str] = []
            for split in splits:
                filenames.extend(by_split[split])
            issues.append(
                LeakageIssue(
                    group_field=field,
                    group_value=value,
                    splits=splits,
                    filenames=tuple(filenames),
                )
            )
    return issues


def infer_sequence_key(filename: str) -> str:
    """Best-effort sequence key from a filename stem.

    ``walk_park_frame_000123.jpg`` and ``walk_park_frame_000124.jpg`` share
    the prefix ``walk_park_frame``.
    """
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem
