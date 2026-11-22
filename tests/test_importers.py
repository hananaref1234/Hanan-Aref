"""Importer adapters on tiny synthetic fixtures. No real datasets."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset.importers import importer_for
from src.dataset.importers.accessibility_barriers import AccessibilityBarriersImporter
from src.dataset.importers.base_importer import ImportError_
from src.dataset.importers.cityscapes import CityscapesImporter
from src.dataset.importers.coverage import coverage_from_records, format_coverage_table
from src.dataset.importers.doordetect import DoorDetectImporter
from src.dataset.importers.download import DownloadBlockedError, assert_download_allowed
from src.dataset.importers.mapillary import MapillaryVistasImporter
from src.dataset.importers.pothole import PotholeImporter
from src.dataset.importers.promote import PromotionError, promote_staging
from src.dataset.importers.provenance import ProvenanceRecord, write_provenance
from src.dataset.importers.review import apply_review_action
from src.dataset.importers.sideguide import SideGuideImporter
from src.dataset.manifest import SourceSpec, get_source
from src.dataset.taxonomy import CUSTOM_CLASSES


def _png(path: Path, color: tuple[int, int, int] = (10, 20, 30), size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = color
    assert cv2.imwrite(str(path), image)


def _source(name: str, raw_dir: Path, **overrides) -> SourceSpec:
    base = get_source(name)
    data = {
        "name": base.name,
        "official_url": base.official_url,
        "license": base.license,
        "license_url": base.license_url,
        "date_accessed": base.date_accessed,
        "download_method": base.download_method,
        "annotation_format": base.annotation_format,
        "source_classes": base.source_classes,
        "listick_mapping": base.listick_mapping,
        "status": base.status,
        "notes": base.notes,
        "raw_dir": str(raw_dir),
        "download_url": base.download_url,
        "request_url": base.request_url,
        "disk_space_hint": base.disk_space_hint,
    }
    data.update(overrides)
    return SourceSpec(**data)


def test_inspect_empty_raw_is_not_present() -> None:
    source = get_source("sideguide")
    report = importer_for(source).inspect()
    assert report.present is False


def test_sideguide_maps_pole_and_not_person_by_default(tmp_path: Path) -> None:
    raw = tmp_path / "sideguide"
    image = raw / "img01.png"
    _png(image)
    (raw / "ann.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "file_name": "img01.png",
                        "width": 32,
                        "height": 32,
                        "split": "train",
                        "sequence_id": "seqA",
                        "boxes": [
                            {"class": "pole", "xtl": 2, "ytl": 2, "xbr": 8, "ybr": 30},
                            {"class": "person", "xtl": 10, "ytl": 10, "xbr": 20, "ybr": 30},
                            {"class": "barricade", "xtl": 21, "ytl": 4, "xbr": 30, "ybr": 16},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    staging = tmp_path / "staging"
    importer = SideGuideImporter(_source("sideguide", raw), staging_root=staging)
    result = importer.convert(include_generic=False)
    classes = {(record.original_class, record.mapped_class, record.proposed_class) for record in result.records}
    assert ("pole", "pole", "pole") in classes
    assert ("barricade", None, "barrier") in {
        (r.original_class, r.mapped_class or None, r.proposed_class or None) for r in result.records
    }
    assert all(record.original_class != "person" for record in result.records)
    assert result.skipped >= 1


def test_accessibility_barriers_maps_semantic_stairs_and_curb(tmp_path: Path) -> None:
    raw = tmp_path / "ab"
    _png(raw / "stair.png")
    xml = """<?xml version="1.0"?>
<annotations>
  <image id="0" name="stair.png" width="32" height="32">
    <box label="stairs" xtl="2" ytl="8" xbr="30" ybr="28">
      <attribute name="steps">4</attribute>
    </box>
    <box label="step" xtl="2" ytl="20" xbr="30" ybr="24">
      <attribute name="type">curb</attribute>
    </box>
    <box label="step" xtl="4" ytl="4" xbr="10" ybr="8">
      <attribute name="type">doorstep</attribute>
    </box>
    <box label="ramp" xtl="12" ytl="12" xbr="20" ybr="18"/>
  </image>
</annotations>
"""
    (raw / "annotations.xml").write_text(xml, encoding="utf-8")
    importer = AccessibilityBarriersImporter(
        _source("accessibility_barriers", raw),
        staging_root=tmp_path / "staging",
    )
    result = importer.convert()
    assert len(result.records) == 4
    by_mapped = {
        (record.original_class, record.mapped_class or None, record.review_status)
        for record in result.records
    }
    assert ("stairs", "stairs", "APPROVED") in by_mapped
    assert ("step", "curb", "APPROVED") in by_mapped
    assert ("step", None, "UNMAPPED_SOURCE_CLASS") in by_mapped
    assert ("ramp", None, "UNMAPPED_SOURCE_CLASS") in by_mapped
    curb = next(
        record
        for record in result.records
        if record.mapped_class == "curb"
    )
    assert curb.source_subtype == "curb"
    assert curb.taxonomy_version == "v0.2"


def test_doordetect_open_stays_pending(tmp_path: Path) -> None:
    raw = tmp_path / "doors" / "Door Classification" / "OriginalSize" / "RGB" / "Train" / "Open"
    _png(raw / "door1.png")
    importer = DoorDetectImporter(
        _source("doordetect", tmp_path / "doors"),
        staging_root=tmp_path / "staging",
    )
    result = importer.convert()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.original_class == "open"
    assert record.proposed_class == "open_doorway"
    assert record.mapped_class == ""
    assert record.source_split == "train"


def test_cityscapes_mask_to_box_and_skips_vegetation(tmp_path: Path) -> None:
    raw = tmp_path / "cs"
    rgb = raw / "leftImg8bit" / "train" / "aa" / "aa_000000_000001_leftImg8bit.png"
    label = raw / "gtFine" / "train" / "aa" / "aa_000000_000001_gtFine_labelIds.png"
    rgb.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.zeros((32, 32), dtype=np.uint8)
    canvas[4:20, 4:8] = 17  # pole
    canvas[2:30, 20:31] = 12  # wall
    canvas[0:4, 0:4] = 21  # vegetation — must not become tree_trunk
    assert cv2.imwrite(str(label), canvas)
    _png(rgb)
    importer = CityscapesImporter(_source("cityscapes", raw), staging_root=tmp_path / "staging")
    result = importer.convert()
    originals = {record.original_class for record in result.records}
    assert "pole" in originals
    assert "wall" in originals
    vegetation = [record for record in result.records if record.original_class == "vegetation"]
    assert all(record.review_status == "UNMAPPED_SOURCE_CLASS" for record in vegetation)
    assert all(record.mapped_class == "" for record in vegetation)
    assert all(record.mapping_type == "MASK_TO_BOX" or record.conversion_method == "mask_to_box" for record in result.records)
    assert all(record.sequence_id == "aa_000000" for record in result.records)
    assert all(record.source_split == "train" for record in result.records)


def test_pothole_refuses_undocumented_source(tmp_path: Path) -> None:
    raw = tmp_path / "pothole"
    _png(raw / "a.png")
    importer = PotholeImporter(_source("pothole", raw), staging_root=tmp_path / "staging")
    with pytest.raises(ImportError_, match="not fully documented"):
        importer.parse_objects()


def test_pothole_maps_named_class_when_documented(tmp_path: Path) -> None:
    raw = tmp_path / "pothole"
    images = raw / "images" / "train"
    labels = raw / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    _png(images / "hole.png")
    (labels / "hole.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("names: [pothole]\n", encoding="utf-8")
    source = _source(
        "pothole",
        raw,
        official_url="https://example.invalid/pothole-cc-by",
        license="CC-BY-4.0",
    )
    importer = PotholeImporter(source, staging_root=tmp_path / "staging")
    result = importer.convert()
    assert len(result.records) == 1
    assert result.records[0].mapped_class == "pothole"
    assert result.records[0].mapping_type == "DIRECT"


def test_mapillary_import_disabled() -> None:
    importer = MapillaryVistasImporter(get_source("mapillary_vistas"))
    with pytest.raises(DownloadBlockedError):
        importer.convert()
    with pytest.raises(DownloadBlockedError):
        assert_download_allowed(get_source("sideguide"))
    with pytest.raises(DownloadBlockedError):
        assert_download_allowed(get_source("mapillary_vistas"))


def test_promote_rejects_sequence_split_leakage(tmp_path: Path) -> None:
    image_a = tmp_path / "stg" / "a.png"
    image_b = tmp_path / "stg" / "b.png"
    _png(image_a)
    _png(image_b)
    records = [
        ProvenanceRecord(
            sample_id="s1",
            source_dataset="sideguide",
            source_image="a.png",
            source_split="train",
            original_class="pole",
            mapped_class="pole",
            mapping_type="DIRECT",
            review_status="APPROVED",
            sequence_id="same_seq",
            license="x",
            original_annotation="xyxy=2,2,8,20;wh=32x32",
            staging_image=str(image_a),
        ),
        ProvenanceRecord(
            sample_id="s2",
            source_dataset="sideguide",
            source_image="b.png",
            source_split="test",
            original_class="pole",
            mapped_class="pole",
            mapping_type="DIRECT",
            review_status="APPROVED",
            sequence_id="same_seq",
            license="x",
            original_annotation="xyxy=2,2,8,20;wh=32x32",
            staging_image=str(image_b),
        ),
    ]
    meta = tmp_path / "staging_metadata.csv"
    write_provenance(meta, records)
    with pytest.raises(PromotionError, match="sequence_id"):
        promote_staging(meta, tmp_path / "dataset")


def test_promote_copies_approved_only(tmp_path: Path) -> None:
    image = tmp_path / "stg" / "pole.png"
    _png(image)
    pending = ProvenanceRecord(
        sample_id="p1",
        source_dataset="sideguide",
        source_image="pole.png",
        source_split="train",
        original_class="pole",
        mapped_class="pole",
        mapping_type="DIRECT",
        review_status="PENDING",
        sequence_id="seq1",
        license="x",
        original_annotation="xyxy=2,2,8,20;wh=32x32",
        staging_image=str(image),
    )
    approved = apply_review_action(pending, "APPROVED")
    meta = tmp_path / "staging_metadata.csv"
    write_provenance(meta, [pending, approved])
    dataset = tmp_path / "dataset"
    summary = promote_staging(meta, dataset)
    assert summary["images"] == 1
    assert (dataset / "images" / "train" / "pole.png").is_file()
    label = (dataset / "labels" / "train" / "pole.txt").read_text(encoding="utf-8").strip()
    assert label.startswith("0 ")


def test_empty_coverage_is_no_data() -> None:
    rows = coverage_from_records([])
    assert [row.name for row in rows] == list(CUSTOM_CLASSES)
    assert all(row.instances == 0 and row.status == "NO_DATA" for row in rows)
    table = format_coverage_table(rows)
    assert "drop_off" in table
    assert "low_overhead_obstacle" in table
    assert "NO_DATA" in table
