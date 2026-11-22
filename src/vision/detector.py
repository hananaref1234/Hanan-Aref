"""Ultralytics YOLO wrapper that emits Li-Stick Detection objects."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from src.dataset.class_map import TrainingClassMap, apply_class_map, load_sidecar_class_map
from src.utils.paths import resolve_from_root
from src.vision.detection import Detection
from src.vision.direction import compute_direction

logger = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Raised when the detection model cannot be loaded or run."""


class ObjectDetector:
    """Load a YOLO checkpoint once and convert results into ``Detection`` objects.

    Callers must not depend on Ultralytics result types. Replacing the backend
    later (for example with NCNN) should only require changes in this module.

    Custom Li-Stick weights use the same API: set ``model_path`` to a trained
    ``.pt`` (or later export). Callers still receive ``Detection`` objects.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        allowed_classes: list[str] | None = None,
        image_size: int = 640,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, got {confidence_threshold}."
            )
        if image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {image_size}.")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = (
            frozenset(allowed_classes) if allowed_classes is not None else None
        )
        self.image_size = image_size
        self._class_map: TrainingClassMap | None = None
        self._model = self._load_model(model_path)

    def detect(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> tuple[list[Detection], float]:
        """Run inference on a BGR or RGB HWC uint8 frame.

        Returns
        -------
        detections, inference_ms
            Filtered ``Detection`` objects and the YOLO call latency in milliseconds.
        """
        self._validate_frame(frame)
        if timestamp is None:
            timestamp = time.time()

        started = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(
                "YOLO inference failed. Check that the frame is a valid image "
                "array and that the model file is compatible with Ultralytics."
            ) from exc
        inference_ms = (time.perf_counter() - started) * 1000.0

        if not results:
            return [], inference_ms

        result = results[0]
        frame_height, frame_width = frame.shape[:2]
        class_names = result.names if getattr(result, "names", None) else {}
        detections: list[Detection] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], inference_ms

        for box in boxes:
            detection = self._box_to_detection(
                box=box,
                class_names=class_names,
                frame_width=frame_width,
                timestamp=timestamp,
            )
            if detection is not None:
                detections.append(detection)
                logger.debug(
                    "Accepted detection: %s",
                    detection.to_dict(),
                )

        return detections, inference_ms

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "The 'ultralytics' package is not installed. "
                "Create a virtual environment and run: pip install -r requirements.txt"
            ) from exc

        resolved = self._resolve_model_path(model_path)
        self._class_map = load_sidecar_class_map(Path(resolved))
        if self._class_map is not None:
            logger.info(
                "Loaded training class map for %s: %s",
                resolved,
                self._class_map.training_to_global(),
            )
        logger.info("Loading YOLO model from %s", resolved)
        try:
            return YOLO(resolved)
        except Exception as exc:
            raise DetectorError(
                f"Could not load YOLO model from {resolved!r}. "
                "If this is a local file, confirm the path exists and is a valid "
                "Ultralytics checkpoint (.pt or exported format). "
                "If this is a pretrained name such as yolo26n.pt, check your "
                "internet connection so Ultralytics can download it."
            ) from exc

    def _resolve_model_path(self, model_path: str) -> str:
        candidate = Path(model_path)
        if candidate.is_absolute():
            if not candidate.exists():
                raise DetectorError(
                    f"Model file not found: {candidate}. "
                    "Update detector.model_path in config/config.yaml."
                )
            return str(candidate)

        from_root = resolve_from_root(candidate)
        if from_root.exists():
            return str(from_root)
        if candidate.parent != Path("."):
            raise DetectorError(
                f"Model file not found: {from_root}. "
                "Update detector.model_path in config/config.yaml."
            )
        # Bare names such as yolo26n.pt are left for Ultralytics to resolve/download.
        return model_path

    def _validate_frame(self, frame: np.ndarray) -> None:
        if frame is None or not isinstance(frame, np.ndarray):
            raise DetectorError("Frame is empty or not a NumPy array.")
        if frame.size == 0:
            raise DetectorError("Frame is empty. The image or camera frame may be corrupt.")
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise DetectorError(
                f"Expected an HWC color frame, got shape {frame.shape}. "
                "Provide a BGR or RGB image from OpenCV or Picamera2."
            )

    def _box_to_detection(
        self,
        box: object,
        class_names: dict[int, str],
        frame_width: int,
        timestamp: float,
    ) -> Detection | None:
        try:
            class_id = int(box.cls[0])  # type: ignore[attr-defined]
            confidence = float(box.conf[0])  # type: ignore[attr-defined]
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("Skipping malformed Ultralytics box: %s", exc)
            return None

        if confidence < self.confidence_threshold:
            return None

        class_name = str(class_names.get(class_id, class_id))
        class_id, class_name = apply_class_map(class_id, class_name, self._class_map)
        if self.allowed_classes is not None and class_name not in self.allowed_classes:
            logger.debug(
                "Filtered class %s (id=%s) confidence=%.3f",
                class_name,
                class_id,
                confidence,
            )
            return None

        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        direction = compute_direction(center_x, frame_width)

        return Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            center_x=center_x,
            center_y=center_y,
            direction=direction,
            timestamp=timestamp,
        )
