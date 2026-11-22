"""Picamera2 backend for Raspberry Pi Camera Module V2 NoIR.

Picamera2 is imported only in this module. On Windows/macOS/Linux laptops
the import fails gracefully until this backend is actually selected.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.camera.base_camera import BaseCamera, CameraError

logger = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - expected on non-Pi development machines
    Picamera2 = None  # type: ignore[misc, assignment]


class Picamera2Error(CameraError):
    """Raised when Picamera2 is unavailable or the Pi camera cannot start."""


class Picamera2Camera(BaseCamera):
    """Capture BGR frames from a Raspberry Pi camera via Picamera2."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._camera = None
        self._rgb_input = True

    def open(self) -> None:
        if Picamera2 is None:
            raise Picamera2Error(
                "Picamera2 is not installed. This backend is for Raspberry Pi OS. "
                "On a laptop, set camera.backend to 'opencv' in config/config.yaml. "
                "On a Raspberry Pi 5: sudo apt install python3-picamera2"
            )
        if self._camera is not None:
            return

        camera = Picamera2()
        video_config = camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        try:
            camera.configure(video_config)
            self._rgb_input = True
        except Exception:
            logger.warning(
                "RGB888 configuration failed; falling back to the Picamera2 default format."
            )
            camera.configure(camera.create_video_configuration(main={"size": (self.width, self.height)}))
            self._rgb_input = False

        try:
            camera.set_controls({"FrameRate": float(self.fps)})
        except Exception as exc:
            logger.warning("Could not set Picamera2 FrameRate to %s: %s", self.fps, exc)

        camera.start()
        self._camera = camera
        logger.info(
            "Started Picamera2 capture %dx%d @ %.1f FPS",
            self.width,
            self.height,
            self.fps,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._camera is None:
            raise Picamera2Error("Camera.read() called before open().")
        try:
            frame = self._camera.capture_array("main")
        except Exception as exc:
            logger.error("Picamera2 capture failed: %s", exc)
            return False, None
        return True, self._to_bgr(frame)

    def close(self) -> None:
        if self._camera is None:
            return
        try:
            self._camera.stop()
        except Exception as exc:
            logger.warning("Picamera2 stop() failed: %s", exc)
        try:
            self._camera.close()
        except Exception as exc:
            logger.warning("Picamera2 close() failed: %s", exc)
        self._camera = None
        logger.info("Released Picamera2 camera")

    def _to_bgr(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3:
            raise Picamera2Error(
                f"Unexpected Picamera2 frame shape {frame.shape}. Expected HWC color."
            )
        channels = frame.shape[2]
        if channels == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if channels == 3 and self._rgb_input:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if channels == 3:
            return frame
        raise Picamera2Error(
            f"Unsupported Picamera2 channel count {channels}. "
            "Configure the camera for RGB888 or XRGB8888."
        )
