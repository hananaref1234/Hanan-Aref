"""Roboflow YOLO importer on synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset.importers.roboflow_yolo import RoboflowYoloImporter
from src.dataset.license_gate import LicenseGateError, assert_import_allowed
from src.dataset.manifest import get_source


def _png(path: Path, color: tuple[int, int, int] = (10, 20, 30), size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = color
    assert cv2.imwrite(str(path), image)


def _source(name: str, raw_dir: Path):
    base = get_source(name)
    return type(base)(
        name=base.name,
        official_url=base.official_url,
        license=base.license,
        license_url=base.license_url,
        date_accessed=base.date_accessed,
        download_method=base.download_method,
        annotation_format=base.annotation_format,
        source_classes=base.source_classes,
        listick_mapping=base.listick_mapping,
        status=base.status,
        notes=base.notes,
        raw_dir=str(raw_dir),
        download_url=base.download_url,
        request_url=base.request_url,
        disk_space_hint=base.disk_space_hint,
        extra=base.extra,
    )


def test_roboflow_imports_bollard_and_review_barricade(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    images = raw / "train" / "images"
    labels = raw / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (raw / "data.yaml").write_text(
        "names:\n  0: bollard\n  1: barricade\n  2: curb\n",
        encoding="utf-8",
    )
    image = images / "a.jpg"
    _png(image)
    (labels / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n1 0.3 0.3 0.2 0.2\n2 0.7 0.7 0.1 0.1\n",
        encoding="utf-8",
    )
    source = _source("roboflow_sidewalk_barriers", raw)
    importer = RoboflowYoloImporter(source, staging_root=tmp_path / "staging")
    result = importer.convert()
    originals = {record.original_class for record in result.records}
    assert "bollard" in originals
    assert "barricade" in originals
    assert "curb" not in originals
    barricade = next(r for r in result.records if r.original_class == "barricade")
    assert barricade.proposed_class == "barrier"
    assert not barricade.mapped_class
    assert "format=bbox" in barricade.original_annotation


def test_roboflow_polygon_records_polygon_to_box_provenance(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    images = raw / "train" / "images"
    labels = raw / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (raw / "data.yaml").write_text(
        "names:\n  0: pole\n  1: tree_trunk\n",
        encoding="utf-8",
    )
    image = images / "a.jpg"
    _png(image)
    (labels / "a.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.6 0.1 0.6\n",
        encoding="utf-8",
    )
    source = _source("roboflow_sidewalk_barriers", raw)
    importer = RoboflowYoloImporter(source, staging_root=tmp_path / "staging")
    result = importer.convert()
    record = result.records[0]
    assert record.original_class == "pole"
    assert record.conversion_method == "POLYGON_TO_BOX"
    assert "format=polygon" in record.original_annotation
    assert "source_label=train" in record.original_annotation


def test_roboflow_tree_class_not_auto_from_open_images_mapping() -> None:
    from src.dataset.importers.mapping import MappingError, assert_auto_mapping_allowed

    with pytest.raises(MappingError):
        assert_auto_mapping_allowed("Tree", "tree_trunk")


def test_sideguide_scottsdale_rejected_for_import() -> None:
    with pytest.raises(LicenseGateError):
        assert_import_allowed(get_source("roboflow_sideguide_scottsdale"))
