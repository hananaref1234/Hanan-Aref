"""Lightweight inventory and duplicate helpers (tiny fixtures only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.dataset.importers.duplicates import find_exact_image_duplicates
from src.dataset.importers.mapping import decide_mapping
from src.dataset.importers.provenance import ProvenanceRecord
from src.dataset.importers.review import class_for_promotion
from src.dataset.importers.source_inventory import inventory_accessibility_barriers
from src.dataset.importers.staging_validate import validate_staging_records
from src.dataset.manifest import get_source


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_inventory_counts_step_types(tmp_path: Path) -> None:
    raw = tmp_path / "ab"
    raw.mkdir()
    xml = """<?xml version="1.0"?>
<annotations>
  <image id="0" name="a.png" width="16" height="16">
    <box label="stairs" xtl="1" ytl="1" xbr="8" ybr="8"/>
    <box label="step" xtl="1" ytl="9" xbr="8" ybr="12">
      <attribute name="type">curb</attribute>
    </box>
    <box label="step" xtl="9" ytl="1" xbr="14" ybr="4">
      <attribute name="type">doorstep</attribute>
    </box>
    <box label="ramp" xtl="9" ytl="9" xbr="14" ybr="14"/>
    <box label="grab bar" xtl="2" ytl="2" xbr="3" ybr="10"/>
  </image>
</annotations>
"""
    (raw / "annotations.xml").write_text(xml, encoding="utf-8")
    _png(raw / "a.png", (1, 2, 3))
    report = inventory_accessibility_barriers(raw)
    assert report.images_in_xml == 1
    assert report.annotations == 5
    assert report.labels["stairs"] == 1
    assert report.labels["step"] == 2
    assert report.labels["ramp"] == 1
    assert report.labels["grab bar"] == 1
    assert report.step_types["curb"] == 1
    assert report.step_types["doorstep"] == 1
    assert report.missing_images == 0


def test_unmapped_is_not_promotable() -> None:
    source = get_source("accessibility_barriers")
    decision = decide_mapping(source, "ramp")
    record = ProvenanceRecord(
        sample_id="u1",
        source_dataset="accessibility_barriers",
        source_image="a.png",
        source_split="",
        original_class="ramp",
        mapped_class="",
        mapping_type=decision.mapping_type,
        review_status=decision.review_status,
        sequence_id="a",
        license="CC-BY-4.0",
    )
    assert record.review_status == "UNMAPPED_SOURCE_CLASS"
    assert class_for_promotion(record) is None


def test_exact_byte_duplicates(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    _png(a, (9, 9, 9))
    _png(b, (9, 9, 9))
    _png(c, (1, 0, 0))
    records = [
        ProvenanceRecord(
            sample_id="1",
            source_dataset="accessibility_barriers",
            source_image="a.png",
            source_split="",
            original_class="stairs",
            mapped_class="",
            mapping_type="MANUAL_DIRECTION",
            review_status="NEEDS_DIRECTION_REVIEW",
            sequence_id="a",
            license="CC-BY-4.0",
            staging_image=str(a),
        ),
        ProvenanceRecord(
            sample_id="2",
            source_dataset="accessibility_barriers",
            source_image="b.png",
            source_split="",
            original_class="stairs",
            mapped_class="",
            mapping_type="MANUAL_DIRECTION",
            review_status="NEEDS_DIRECTION_REVIEW",
            sequence_id="b",
            license="CC-BY-4.0",
            staging_image=str(b),
        ),
        ProvenanceRecord(
            sample_id="3",
            source_dataset="accessibility_barriers",
            source_image="c.png",
            source_split="",
            original_class="stairs",
            mapped_class="",
            mapping_type="MANUAL_DIRECTION",
            review_status="NEEDS_DIRECTION_REVIEW",
            sequence_id="c",
            license="CC-BY-4.0",
            staging_image=str(c),
        ),
    ]
    groups = find_exact_image_duplicates(records)
    assert len(groups) == 1
    assert set(groups[0].source_images) == {"a.png", "b.png"}
    assert "c.png" not in groups[0].source_images


def test_staging_validate_catches_invented_direction(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    _png(image, (3, 3, 3))
    bad = ProvenanceRecord(
        sample_id="dup",
        source_dataset="accessibility_barriers",
        source_image="a.png",
        source_split="",
        original_class="stair",
        mapped_class="stairs_up",
        mapping_type="MANUAL_DIRECTION",
        review_status="NEEDS_DIRECTION_REVIEW",
        sequence_id="a",
        license="CC-BY-4.0",
        original_annotation="xyxy=1,1,8,8;wh=16x16",
        staging_image=str(image),
    )
    issues = validate_staging_records([bad, bad])
    assert any("Duplicate sample_id" in item for item in issues)
    assert any("already has mapped_class" in item for item in issues)
