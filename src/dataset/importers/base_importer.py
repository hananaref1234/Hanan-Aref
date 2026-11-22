"""Shared importer interface. Source-specific logic lives in sibling modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.importers.boxes import PixelBox, xyxy_to_yolo
from src.dataset.importers.mapping import MappingDecision, decide_mapping
from src.dataset.importers.provenance import ProvenanceRecord, make_sample_id
from src.dataset.manifest import SourceSpec
from src.dataset.taxonomy import CLASS_TO_ID, TAXONOMY_VERSION
from src.utils.paths import project_root


class ImportError_(RuntimeError):
    """Raised when a source cannot be imported safely."""


@dataclass
class SourceInspectReport:
    source_name: str
    raw_dir: Path
    present: bool
    annotation_files: int = 0
    image_files: int = 0
    notes: str = ""
    detected_format: str = ""


@dataclass
class ParsedObject:
    image_path: Path
    source_image: str
    source_split: str
    sequence_id: str
    original_class: str
    box: PixelBox
    image_width: int
    image_height: int
    attributes: dict[str, str] = field(default_factory=dict)
    original_annotation: str = ""
    conversion_method: str = "bbox"


@dataclass
class ImportResult:
    records: list[ProvenanceRecord] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


class BaseImporter(ABC):
    """Inspect a raw source and emit staging provenance records."""

    source_name: str

    def __init__(self, source: SourceSpec, staging_root: Path | None = None) -> None:
        self.source = source
        self.staging_root = staging_root or (project_root() / "data" / "staging")

    @abstractmethod
    def inspect(self) -> SourceInspectReport:
        """Describe what is on disk without converting."""

    @abstractmethod
    def parse_objects(self) -> list[ParsedObject]:
        """Read original annotations. Must not invent Li-Stick labels."""

    def convert(self, *, include_generic: bool = False) -> ImportResult:
        """Map safe classes and write staging images/labels plus provenance."""
        report = self.inspect()
        if not report.present:
            raise ImportError_(
                f"Raw data for {self.source.name} is not present at {report.raw_dir}. "
                "See docs/DATASET_ACQUISITION.md."
            )
        result = ImportResult()
        parsed = self.parse_objects()
        by_image: dict[Path, list[tuple[ParsedObject, MappingDecision, int]]] = {}
        for parsed_object in parsed:
            decision = decide_mapping(
                self.source,
                parsed_object.original_class,
                extra_attributes=parsed_object.attributes,
            )
            if decision.taxonomy == "combined" and not include_generic:
                result.skipped += 1
                continue
            keep_statuses = {
                "NEEDS_DIRECTION_REVIEW",
                "PENDING",
                "UNMAPPED_SOURCE_CLASS",
                "APPROVED",
            }
            if (
                decision.mapped_class is None
                and decision.proposed_class is None
                and decision.review_status not in keep_statuses
            ):
                result.skipped += 1
                continue
            box_index = len(by_image.setdefault(parsed_object.image_path, []))
            by_image[parsed_object.image_path].append((parsed_object, decision, box_index))

        for image_path, items in by_image.items():
            needs_copy = any(
                decision.review_status == "NEEDS_DIRECTION_REVIEW"
                or decision.mapped_class
                or decision.proposed_class
                for _parsed, decision, _index in items
            )
            if needs_copy:
                try:
                    staging_image, staging_label = self._stage_image(
                        image_path,
                        items[0][0].source_split,
                    )
                except OSError as exc:
                    result.errors.append(f"{image_path}: {exc}")
                    continue
            else:
                staging_image = image_path
                staging_label = Path("")
            label_lines: list[str] = []
            for parsed_object, decision, box_index in items:
                sample_id = make_sample_id(
                    self.source.name,
                    parsed_object.source_split,
                    image_path.stem,
                    box_index,
                )
                mapped_id = None
                class_name = decision.mapped_class or ""
                if class_name in CLASS_TO_ID:
                    yolo = xyxy_to_yolo(
                        parsed_object.box,
                        parsed_object.image_width,
                        parsed_object.image_height,
                        CLASS_TO_ID[class_name],
                    )
                    # Only write YOLO lines for classes already assigned (DIRECT).
                    # Directional/semantic review samples stay in provenance until approved.
                    if decision.review_status != "NEEDS_DIRECTION_REVIEW" and decision.mapped_class:
                        label_lines.append(
                            f"{yolo.class_id} {yolo.x_center:.6f} {yolo.y_center:.6f} "
                            f"{yolo.width:.6f} {yolo.height:.6f}"
                        )
                        mapped_id = yolo.class_id
                record = ProvenanceRecord(
                    sample_id=sample_id,
                    source_dataset=self.source.name,
                    source_image=parsed_object.source_image,
                    source_split=parsed_object.source_split,
                    original_class=parsed_object.original_class,
                    mapped_class=decision.mapped_class or "",
                    mapping_type=decision.mapping_type,
                    review_status=decision.review_status,
                    sequence_id=parsed_object.sequence_id,
                    license=self.source.license,
                    proposed_class=decision.proposed_class or "",
                    taxonomy=decision.taxonomy,
                    conversion_method=parsed_object.conversion_method,
                    original_annotation=_merge_annotation(
                        parsed_object.original_annotation,
                        parsed_object,
                    ),
                    staging_image=str(staging_image),
                    staging_label=str(staging_label),
                    notes=decision.notes,
                    taxonomy_version=TAXONOMY_VERSION,
                    source_subtype=str(parsed_object.attributes.get("type", "") or ""),
                )
                _ = mapped_id
                result.records.append(record)
            if staging_label.name:
                staging_label.write_text(
                    ("\n".join(label_lines) + ("\n" if label_lines else "")),
                    encoding="utf-8",
                )
        return result

    def _stage_image(self, image_path: Path, source_split: str) -> tuple[Path, Path]:
        from src.dataset.importers.staging import copy_to_staging

        return copy_to_staging(
            image_path,
            self.staging_root,
            source_name=self.source.name,
            source_split=source_split or "unsplit",
        )


def _format_box(parsed_object: ParsedObject) -> str:
    box = parsed_object.box
    return (
        f"xyxy={box.x1:.1f},{box.y1:.1f},{box.x2:.1f},{box.y2:.1f};"
        f"wh={parsed_object.image_width}x{parsed_object.image_height}"
    )


def _merge_annotation(original: str, parsed_object: ParsedObject) -> str:
    geometry = _format_box(parsed_object)
    if not original:
        return geometry
    if "xyxy=" in original:
        return original
    return f"{geometry};{original}"


def raw_has_payload(raw_dir: Path) -> bool:
    """True when the raw folder contains more than README/placeholder files."""
    if not raw_dir.is_dir():
        return False
    skip_names = {".gitkeep", "readme.md", "readme.txt"}
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.name.lower() not in skip_names:
            return True
    return False
