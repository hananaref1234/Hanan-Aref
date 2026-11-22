"""Application logging setup."""

from __future__ import annotations

import logging

_VALID_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once for CLI entry points.

    Parameters
    ----------
    level:
        Logging level name such as ``INFO`` or ``DEBUG``.
    """
    normalized = level.upper()
    if normalized not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid logging level {level!r}. "
            f"Use one of: {', '.join(sorted(_VALID_LEVELS))}."
        )

    logging.basicConfig(
        level=getattr(logging, normalized),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
