"""Dataset statistics for class balance and coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.taxonomy import CUSTOM_CLASSES, SAFETY_CRITICAL_CLASSES
from src.dataset.validator import IMAGE_SUFFIXES, SPLITS
from src.dataset.yolo_labels import parse_yolo_label_file

LOW_SAMPLE_WARNING = 200
SEVERE_SAMPLE_WARNING = 100


@dataclass
class DatasetStats:
    split_image_counts: dict[str, int] = field(default_factory=dict)
    instances_per_class: dict[str, int] = field(default_factory=dict)
    images_containing_class: dict[str, int] = field(default_factory=dict)
    average_objects_per_image: float = 0.0
    box_area_sum: dict[str, float] = field(default_factory=dict)
    box_count: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        total_images = sum(self.split_image_counts.values())
        total_instances = sum(self.instances_per_class.values())
        lines = [
            "Dataset Statistics",
            "------------------",
            f"Images: {total_images}",
            f"  train: {self.split_image_counts.get('train', 0)}",
            f"  val:   {self.split_image_counts.get('val', 0)}",
            f"  test:  {self.split_image_counts.get('test', 0)}",
            f"Labeled instances: {total_instances}",
            f"Average objects per image: {self.average_objects_per_image:.2f}",
            "",
            "Instances per class",
        ]
        if total_images == 0:
            lines.append("  (no images yet)")
            lines.append("")
            lines.append("CUSTOM DATASET REQUIRED")
            return "\n".join(lines)

        max_count = max(self.instances_per_class.values()) if total_instances else 0
        for name in CUSTOM_CLASSES:
            count = self.instances_per_class.get(name, 0)
            images_with = self.images_containing_class.get(name, 0)
            percent = 100.0 * images_with / total_images if total_images else 0.0
            warning = ""
            if count == 0:
                warning = "  <-- WARNING: no samples"
            elif count < SEVERE_SAMPLE_WARNING:
                warning = "  <-- WARNING: low sample count"
            elif count < LOW_SAMPLE_WARNING:
                warning = "  <-- WARNING: underrepresented"
            if max_count and count > 0 and max_count / max(count, 1) >= 10:
                warning = warning or "  <-- WARNING: class imbalance vs majority class"
            avg_area = (
                self.box_area_sum.get(name, 0.0) / self.box_count[name]
                if self.box_count.get(name)
                else 0.0
            )
            lines.append(
                f"  {name:22} {count:5d}  images={percent:5.1f}%  "
                f"mean_box_area={avg_area:.4f}{warning}"
            )

        lines.append("")
        lines.append("Safety-critical classes")
        for name in SAFETY_CRITICAL_CLASSES:
            count = self.instances_per_class.get(name, 0)
            lines.append(f"  {name}: {count}")
        lines.append("")
        lines.append("Diversity matters more than raw image count.")
        lines.append("Near-duplicate walking-video frames should not be counted as unique examples.")
        return "\n".join(lines)


def compute_dataset_stats(dataset_root: Path) -> DatasetStats:
    dataset_root = dataset_root.resolve()
    stats = DatasetStats(
        instances_per_class={name: 0 for name in CUSTOM_CLASSES},
        images_containing_class={name: 0 for name in CUSTOM_CLASSES},
        box_area_sum={name: 0.0 for name in CUSTOM_CLASSES},
        box_count={name: 0 for name in CUSTOM_CLASSES},
    )
    total_objects = 0
    total_images = 0
    max_class_id = len(CUSTOM_CLASSES) - 1

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = []
        if image_dir.exists():
            images = [
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
        stats.split_image_counts[split] = len(images)
        total_images += len(images)
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            try:
                boxes = parse_yolo_label_file(label_path, max_class_id=max_class_id)
            except Exception:
                continue
            present: set[str] = set()
            for box in boxes:
                name = CUSTOM_CLASSES[box.class_id]
                stats.instances_per_class[name] += 1
                stats.box_area_sum[name] += box.width * box.height
                stats.box_count[name] += 1
                present.add(name)
                total_objects += 1
            for name in present:
                stats.images_containing_class[name] += 1

    stats.average_objects_per_image = (
        total_objects / total_images if total_images else 0.0
    )
    return stats
