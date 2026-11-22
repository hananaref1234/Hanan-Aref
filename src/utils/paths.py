"""Project-root path helpers.

Scripts and tests may be launched from any working directory, so configuration
and model paths are resolved from the repository root rather than cwd.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (the directory that contains ``src/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_from_root(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate
