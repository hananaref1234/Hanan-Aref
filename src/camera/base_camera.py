"""Abstract camera interface shared by OpenCV and Picamera2 backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

import numpy as np


class CameraError(RuntimeError):
    """Raised when a camera backend cannot open, capture, or shut down cleanly."""


class BaseCamera(ABC):
    """Minimal capture interface used by the live and recorded-video loops."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the capture device or file."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return ``(True, BGR frame)`` or ``(False, None)`` on failure / EOF."""

    @abstractmethod
    def close(self) -> None:
        """Release hardware or file handles."""

    def __enter__(self) -> BaseCamera:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
