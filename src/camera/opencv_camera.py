"""OpenCV capture backend for webcams and video files."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.camera.base_camera import BaseCamera, CameraError

logger = logging.getLogger(__name__)


class OpenCVCamera(BaseCamera):
    """Capture BGR frames from a USB webcam, device index, or video file."""

    def __init__(
        self,
        device: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: cv2.VideoCapture | None = None

    @property
    def is_file_source(self) -> bool:
        return isinstance(self.device, str)

    def open(self) -> None:
        if self._capture is not None:
            return

        if self.is_file_source:
            source_path = Path(str(self.device))
            if not source_path.exists():
                raise CameraError(
                    f"Video file not found: {source_path}. "
                    "Pass an existing .mp4/.avi path to scripts/test_video.py."
                )

        capture = cv2.VideoCapture(self.device)
        if not capture.isOpened():
            raise CameraError(
                f"Could not open OpenCV camera source {self.device!r}. "
                "On a laptop, confirm a webcam is connected and camera.device "
                "in config/config.yaml is the correct index (often 0). "
                "For a video file, confirm the path is readable."
            )

        if not self.is_file_source:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            capture.set(cv2.CAP_PROP_FPS, float(self.fps))

        self._capture = capture
        logger.info(
            "Opened OpenCV camera source=%r requested=%dx%d @ %.1f FPS",
            self.device,
            self.width,
            self.height,
            self.fps,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is None:
            raise CameraError("Camera.read() called before open().")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def close(self) -> None:
        if self._capture is None:
            return
        self._capture.release()
        self._capture = None
        logger.info("Released OpenCV camera source=%r", self.device)
