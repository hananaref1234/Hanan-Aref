"""Select a camera backend from configuration."""

from __future__ import annotations

from src.camera.base_camera import BaseCamera
from src.camera.opencv_camera import OpenCVCamera
from src.utils.config_loader import CameraConfig


def create_camera(config: CameraConfig) -> BaseCamera:
    """Instantiate the backend named in ``config.backend``."""
    if config.backend == "opencv":
        return OpenCVCamera(
            device=config.device,
            width=config.width,
            height=config.height,
            fps=config.fps,
        )
    if config.backend == "picamera2":
        from src.camera.picamera2_camera import Picamera2Camera

        return Picamera2Camera(
            width=config.width,
            height=config.height,
            fps=config.fps,
        )
    raise ValueError(
        f"Unknown camera backend {config.backend!r}. "
        "Use 'opencv' (laptop/webcam/video) or 'picamera2' (Raspberry Pi)."
    )
