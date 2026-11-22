"""Source manifest validation. Does not download datasets."""

from __future__ import annotations

import pytest

from src.dataset.manifest import (
    ALLOWED_STATUSES,
    ManifestError,
    get_source,
    load_manifest,
)


def test_manifest_loads_and_has_required_fields() -> None:
    sources = {source.name: source for source in load_manifest()}
    expected = {
        "sideguide",
        "accessibility_barriers",
        "doordetect",
        "cityscapes",
        "pothole",
        "mapillary_vistas",
        "listick_camera",
        "open_images_v7",
        "timbervision",
        "roboflow_sidewalk_barriers",
        "roboflow_bollard_v2gn5",
        "roboflow_power_pole",
        "roboflow_bollard_nkhwt",
        "roboflow_asset_model",
        "roboflow_bollards_crosswalk_stairs",
        "roboflow_sideguide_scottsdale",
    }
    assert expected <= set(sources)
    for source in sources.values():
        assert source.status in ALLOWED_STATUSES
        assert source.status != "APPROVED"
        assert source.name
        assert source.date_accessed
        assert source.notes


def test_no_source_is_approved_before_verification() -> None:
    assert all(source.status != "APPROVED" for source in load_manifest())


def test_sideguide_remains_manual_download() -> None:
    source = get_source("sideguide")
    assert source.status == "MANUAL_DOWNLOAD_REQUIRED"
    assert source.download_method == "request_form"
    assert source.request_url
    assert "noncommercial" in source.license or "non-commercial" in source.notes.lower()


def test_mapillary_is_license_review_only() -> None:
    source = get_source("mapillary_vistas")
    assert source.status == "LICENSE_REVIEW_REQUIRED"
    assert source.download_method == "disabled"


def test_accessibility_barriers_license_is_cc_by() -> None:
    source = get_source("accessibility_barriers")
    assert "CC-BY-4.0" in source.license
    assert source.status == "MANUAL_REVIEW_REQUIRED"


def test_pothole_is_unselected() -> None:
    source = get_source("pothole")
    assert source.official_url == ""
    assert source.license == ""
    assert source.status == "MANUAL_REVIEW_REQUIRED"


def test_unknown_source_raises() -> None:
    with pytest.raises(ManifestError, match="Unknown source"):
        get_source("not_a_dataset")
