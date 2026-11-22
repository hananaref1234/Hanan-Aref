"""License gate enforcement."""

from __future__ import annotations

import pytest

from src.dataset.license_gate import LicenseGateError, assert_import_allowed
from src.dataset.manifest import get_source


def test_sideguide_import_blocked() -> None:
    source = get_source("sideguide")
    with pytest.raises(LicenseGateError, match="not approved"):
        assert_import_allowed(source)


def test_open_images_rejected() -> None:
    source = get_source("open_images_v7")
    with pytest.raises(LicenseGateError, match="REJECTED"):
        assert_import_allowed(source)


def test_timbervision_license_review_blocked() -> None:
    source = get_source("timbervision")
    with pytest.raises(LicenseGateError, match="LICENSE_REVIEW_REQUIRED"):
        assert_import_allowed(source)


def test_roboflow_sidewalk_barriers_allowed_when_documented() -> None:
    source = get_source("roboflow_sidewalk_barriers")
    assert source.license.startswith("CC-BY")
    assert_import_allowed(source)


def test_roboflow_unclear_license_blocked() -> None:
    source = get_source("roboflow_bollard_nkhwt")
    with pytest.raises(LicenseGateError, match="LICENSE_REVIEW_REQUIRED"):
        assert_import_allowed(source)
