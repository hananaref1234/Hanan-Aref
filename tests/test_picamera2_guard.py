"""Picamera2 must import on machines that do not have the Pi camera stack."""

from __future__ import annotations

import pytest

from src.camera.picamera2_camera import Picamera2, Picamera2Camera, Picamera2Error


def test_picamera2_backend_is_importable() -> None:
    camera = Picamera2Camera(width=1280, height=720, fps=30)
    if Picamera2 is None:
        with pytest.raises(Picamera2Error, match="Picamera2 is not installed"):
            camera.open()
    else:
        camera.close()
