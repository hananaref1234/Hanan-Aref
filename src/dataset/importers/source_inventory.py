"""Count original Accessibility Barriers CVAT labels without mapping them."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.importers.cvat import parse_cvat_xml
from src.dataset.importers.mapping import normalize_class_name
from src.dataset.importers.staging import IMAGE_SUFFIXES


@dataclass
class AccessibilityInventory:
    xml_files: int = 0
    images_in_xml: int = 0
    images_on_disk: int = 0
    missing_images: int = 0
    annotations: int = 0
    labels: Counter = field(default_factory=Counter)
    step_types: Counter = field(default_factory=Counter)
    other_attributes: dict[str, Counter] = field(default_factory=dict)


def inventory_accessibility_barriers(raw_dir: Path) -> AccessibilityInventory:
    """Parse CVAT XML under raw_dir and count labels/attributes as they are."""
    report = AccessibilityInventory()
    xml_files = sorted(raw_dir.rglob("*.xml")) if raw_dir.is_dir() else []
    report.xml_files = len(xml_files)
    report.images_on_disk = (
        sum(1 for path in raw_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if raw_dir.is_dir()
        else 0
    )
    xml_names: set[str] = set()
    for xml_path in xml_files:
        for image in parse_cvat_xml(xml_path):
            xml_names.add(Path(image.name).name)
            report.images_in_xml += 1
            resolved = _resolve(raw_dir, image.name)
            if not resolved.is_file():
                report.missing_images += 1
            for box in image.boxes:
                report.annotations += 1
                label = box.label.strip()
                report.labels[label] += 1
                for key, value in box.attributes.items():
                    key_norm = normalize_class_name(key)
                    if key_norm == "type" and normalize_class_name(label) == "step":
                        report.step_types[value.strip() or "<empty>"] += 1
                    else:
                        report.other_attributes.setdefault(f"{label}.{key}", Counter())[value] += 1
    return report


def _resolve(raw_dir: Path, name: str) -> Path:
    direct = raw_dir / name
    if direct.is_file():
        return direct
    matches = list(raw_dir.rglob(Path(name).name))
    return matches[0] if matches else direct
