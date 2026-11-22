"""Class-coverage report from staging provenance (zeros if nothing imported)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.dataset.importers.provenance import ProvenanceRecord, read_provenance
from src.dataset.importers.review import class_for_promotion
from src.dataset.taxonomy import CUSTOM_CLASSES, SAFETY_CRITICAL_CLASSES, TAXONOMY_VERSION

STATUS_GOOD = "GOOD"
STATUS_NEEDS_MORE = "NEEDS_MORE_DATA"
STATUS_NO_DATA = "NO_DATA"
STATUS_WEAK = "WEAK"

LOW_APPROVED = 200


@dataclass(frozen=True)
class ClassCoverage:
    name: str
    instances: int
    images: int
    sources: tuple[str, ...]
    pending_review: int
    status: str
    weak: bool = False


def coverage_from_records(records: list[ProvenanceRecord]) -> list[ClassCoverage]:
    counts: dict[str, int] = defaultdict(int)
    images: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    pending: dict[str, int] = defaultdict(int)

    for record in records:
        detector_class = class_for_promotion(record)
        if detector_class:
            counts[detector_class] += 1
            images[detector_class].add(record.source_image)
            sources[detector_class].add(record.source_dataset)
            continue
        proposed = record.mapped_class or record.proposed_class
        if proposed in CUSTOM_CLASSES and record.review_status in {
            "PENDING",
            "NEEDS_DIRECTION_REVIEW",
        }:
            pending[proposed] += 1
            sources[proposed].add(record.source_dataset)

    rows: list[ClassCoverage] = []
    for name in CUSTOM_CLASSES:
        instance_count = counts[name]
        status = _status(instance_count)
        rows.append(
            ClassCoverage(
                name=name,
                instances=instance_count,
                images=len(images[name]),
                sources=tuple(sorted(sources[name])),
                pending_review=pending[name],
                status=status,
                weak=status in {STATUS_WEAK, STATUS_NO_DATA} and name in {
                    "drop_off",
                    "low_overhead_obstacle",
                    "open_doorway",
                },
            )
        )
    return rows


def coverage_from_staging(staging_metadata: Path) -> list[ClassCoverage]:
    return coverage_from_records(read_provenance(staging_metadata))


def format_coverage_table(rows: list[ClassCoverage]) -> str:
    lines = [
        "CLASS                    IMAGES       INSTANCES    SOURCES                  STATUS",
        "",
    ]
    for row in rows:
        source_text = ",".join(row.sources) if row.sources else "-"
        weak_mark = "  <-- WEAK" if row.status == STATUS_WEAK else ""
        lines.append(
            f"{row.name:<24} {row.images:<12} {row.instances:<12} "
            f"{source_text:<24} {row.status}{weak_mark}"
        )
    return "\n".join(lines) + "\n"


def format_coverage_markdown(rows: list[ClassCoverage]) -> str:
    populated = [row.name for row in rows if row.instances > 0]
    empty = [row.name for row in rows if row.instances == 0]
    safety = [
        name
        for name in SAFETY_CRITICAL_CLASSES
        if any(row.name == name and row.instances == 0 for row in rows)
    ]
    populated_text = ", ".join(f"`{name}`" for name in populated) or "(none)"
    empty_text = ", ".join(f"`{name}`" for name in empty) or "(none)"
    safety_text = ", ".join(f"`{name}`" for name in safety) or "(none)"
    return (
        f"# Li-Stick custom class coverage (taxonomy {TAXONOMY_VERSION})\n\n"
        "Generated from staging provenance. Instance counts are semantic "
        "detector labels (`stairs`, `curb`, …). Direction is not a detector "
        "class in v0.2. Empty classes are reported as zero; they are not "
        "fabricated.\n\n"
        "**Do not train until this snapshot is reviewed.**\n\n"
        "## Coverage table\n\n"
        "```text\n"
        f"{format_coverage_table(rows)}"
        "```\n\n"
        f"**Populated classes:** {populated_text}\n\n"
        f"**NO_DATA classes:** {empty_text}\n\n"
        f"**Safety-critical classes still empty:** {safety_text}\n\n"
        "Direction review is not required for v0.2 object-detector training. "
        "Preserved v0.1 `STAIRS_*` / `CURB_*` values remain in staging "
        "`direction_status` / `direction_class` for later geometry research.\n"
    )


def _status(instances: int) -> str:
    if instances <= 0:
        return STATUS_NO_DATA
    if instances < LOW_APPROVED:
        return STATUS_WEAK
    if instances < 500:
        return STATUS_NEEDS_MORE
    return STATUS_GOOD
