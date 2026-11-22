"""Provenance CSV round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.importers.provenance import (
    PROVENANCE_FIELDS,
    ProvenanceRecord,
    make_sample_id,
    read_provenance,
    write_provenance,
)


def test_provenance_round_trip(tmp_path: Path) -> None:
    record = ProvenanceRecord(
        sample_id=make_sample_id("sideguide", "train", "img01", 0),
        source_dataset="sideguide",
        source_image="img01.jpg",
        source_split="train",
        original_class="pole",
        mapped_class="pole",
        mapping_type="DIRECT",
        review_status="PENDING",
        sequence_id="walk01",
        license="unknown_access_controlled",
        proposed_class="pole",
        conversion_method="bbox",
        original_annotation="xyxy=1,2,3,4;wh=64x64",
        staging_image="a.jpg",
        staging_label="a.txt",
    )
    path = tmp_path / "staging_metadata.csv"
    write_provenance(path, [record])
    loaded = read_provenance(path)
    assert len(loaded) == 1
    assert loaded[0].sample_id == "sideguide_train_img01_0000"
    assert loaded[0].source_dataset == "sideguide"
    assert loaded[0].mapping_type == "DIRECT"
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    for field in ("sample_id", "source_dataset", "source_image", "mapped_class", "review_status", "license"):
        assert field in header
    assert list(PROVENANCE_FIELDS[:10]) == [
        "sample_id",
        "source_dataset",
        "source_image",
        "source_split",
        "original_class",
        "mapped_class",
        "mapping_type",
        "review_status",
        "sequence_id",
        "license",
    ]


def test_invalid_mapping_type_rejected(tmp_path: Path) -> None:
    record = ProvenanceRecord(
        sample_id="x",
        source_dataset="sideguide",
        source_image="a.jpg",
        source_split="train",
        original_class="pole",
        mapped_class="pole",
        mapping_type="NOT_A_TYPE",
        review_status="PENDING",
        sequence_id="s",
        license="x",
    )
    with pytest.raises(ValueError, match="Invalid mapping_type"):
        write_provenance(tmp_path / "bad.csv", [record])
