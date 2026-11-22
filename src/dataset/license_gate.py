"""License and approval gates for dataset import and download.

No source may be marked APPROVED in the manifest until a human verifies
license text and class semantics. Importers call these helpers before
touching third-party data.
"""

from __future__ import annotations

from src.dataset.manifest import SourceSpec

IMPORT_BLOCKED_STATUSES = frozenset({"REJECTED", "LICENSE_REVIEW_REQUIRED"})

# SideGuide is recorded for traceability but must not enter Li-Stick staging.
LISTICK_BLOCKED_SOURCES = frozenset({"sideguide"})


class LicenseGateError(RuntimeError):
    """Raised when a source cannot be imported or downloaded safely."""


def assert_import_allowed(source: SourceSpec) -> None:
    """Refuse staging conversion for blocked or unapproved sources."""
    if source.name in LISTICK_BLOCKED_SOURCES:
        raise LicenseGateError(
            f"Source {source.name!r} is not approved for Li-Stick use. "
            "Do not import until access and licensing are explicitly approved."
        )
    if source.status in IMPORT_BLOCKED_STATUSES:
        raise LicenseGateError(
            f"Refusing to import {source.name!r} with status {source.status}. "
            "Resolve license review or rejection before import."
        )


def import_status_allows(source: SourceSpec) -> bool:
    """Return True when import is permitted for a source."""
    try:
        assert_import_allowed(source)
    except LicenseGateError:
        return False
    return True
