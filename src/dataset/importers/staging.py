"""Copy or link source images into staging without touching the final dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def copy_to_staging(
    image_path: Path,
    staging_root: Path,
    *,
    source_name: str,
    source_split: str,
    link: bool = False,
) -> tuple[Path, Path]:
    """Copy one image into staging and return (image_path, label_path)."""
    split = source_split or "unsplit"
    image_dir = staging_root / "images" / source_name / split
    label_dir = staging_root / "labels" / source_name / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    dest_image = image_dir / image_path.name
    dest_label = label_dir / f"{image_path.stem}.txt"
    if not dest_image.exists():
        if link:
            try:
                dest_image.hardlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, dest_image)
        else:
            shutil.copy2(image_path, dest_image)
    return dest_image, dest_label


def infer_sequence_id(path: Path, fallback: str = "") -> str:
    """Prefer a parent folder, else a numeric-suffix-stripped stem."""
    parent = path.parent.name
    if parent and parent not in {"images", "img", "rgb", "train", "val", "test"}:
        return f"{parent}/{path.stem}"
    stem = path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return fallback or stem
