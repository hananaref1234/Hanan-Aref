"""Object tracker interface and ByteTrack wrapper.

Ultralytics ByteTrack is used only inside this module. Callers receive
``TrackedDetection`` objects and never see tracker-library types.

ByteTrack's ``model.track()`` API would re-run YOLO and couple tracking to
the detector, so this wrapper feeds our existing ``Detection`` objects into
``BYTETracker.update`` instead.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np

from src.utils.config_loader import TrackingConfig
from src.tracking.tracked_detection import TrackedDetection
from src.vision.detection import Detection
from src.vision.direction import compute_direction

logger = logging.getLogger(__name__)


class TrackerError(RuntimeError):
    """Raised when the tracker backend cannot be constructed or updated."""


class BaseTracker(ABC):
    """Replaceable tracking interface used by the live loop."""

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        """Associate this frame's detections with persistent track IDs."""


class ObjectTracker(BaseTracker):
    """ByteTrack-backed tracker that consumes Phase 1 ``Detection`` objects."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._backend = self._build_bytetrack(config)

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        frame_width = int(frame.shape[1]) if frame is not None else _infer_frame_width(detections)
        batch = _DetectionBatch.from_detections(detections)
        try:
            tracks = self._backend.update(batch, img=frame)
        except Exception as exc:
            raise TrackerError(
                "ByteTrack failed to update. Check that detections have valid "
                "bounding boxes and that the ultralytics tracker API is available."
            ) from exc

        if tracks is None or len(tracks) == 0:
            return []

        tracked: list[TrackedDetection] = []
        for row in np.atleast_2d(tracks):
            converted = _row_to_tracked(row, detections, frame_width)
            if converted is not None:
                tracked.append(converted)
        return tracked

    def _build_bytetrack(self, config: TrackingConfig):
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:
            raise TrackerError(
                "Ultralytics ByteTrack is not available. Install project "
                "requirements with: pip install -r requirements.txt"
            ) from exc

        args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=config.max_missing_frames,
            match_thresh=0.8,
            fuse_score=True,
        )
        logger.info(
            "Initialized ByteTrack (max_missing_frames=%s)",
            config.max_missing_frames,
        )
        return BYTETracker(args)


def create_tracker(config: TrackingConfig) -> BaseTracker:
    """Factory so the live loop does not import a specific tracker class."""
    if config.tracker != "bytetrack":
        raise TrackerError(
            f"Unsupported tracker {config.tracker!r}. Phase 2 supports 'bytetrack'."
        )
    return ObjectTracker(config)


class _DetectionBatch:
    """Minimal Results-like object for Ultralytics BYTETracker.update()."""

    def __init__(
        self,
        xyxy: np.ndarray,
        xywh: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = xyxy
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> _DetectionBatch:
        if not detections:
            return cls(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                xywh=np.zeros((0, 4), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.float32),
            )
        xyxy = np.array(
            [[det.x1, det.y1, det.x2, det.y2] for det in detections],
            dtype=np.float32,
        )
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        xywh = np.stack(
            [
                xyxy[:, 0] + widths / 2.0,
                xyxy[:, 1] + heights / 2.0,
                widths,
                heights,
            ],
            axis=1,
        ).astype(np.float32)
        conf = np.array([det.confidence for det in detections], dtype=np.float32)
        class_ids = np.array([det.class_id for det in detections], dtype=np.float32)
        return cls(xyxy=xyxy, xywh=xywh, conf=conf, cls=class_ids)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, index: object) -> _DetectionBatch:
        return _DetectionBatch(
            xyxy=np.atleast_2d(self.xyxy[index]),
            xywh=np.atleast_2d(self.xywh[index]),
            conf=np.atleast_1d(self.conf[index]),
            cls=np.atleast_1d(self.cls[index]),
        )


def _row_to_tracked(
    row: np.ndarray,
    detections: list[Detection],
    frame_width: int,
) -> TrackedDetection | None:
    if len(row) < 8:
        return None
    x1, y1, x2, y2 = (int(round(float(v))) for v in row[:4])
    track_id = int(row[4])
    score = float(row[5])
    class_id = int(row[6])
    source_index = int(row[7])

    source: Detection | None = None
    if 0 <= source_index < len(detections):
        source = detections[source_index]

    class_name = source.class_name if source is not None else str(class_id)
    timestamp = source.timestamp if source is not None else 0.0
    if source is not None and source.class_id == class_id:
        class_id = source.class_id

    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))
    if frame_width <= 0:
        frame_width = max(x2, 1)
    direction = compute_direction(center_x, frame_width)

    return TrackedDetection(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=score,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=center_x,
        center_y=center_y,
        direction=direction,
        timestamp=timestamp,
    )


def _infer_frame_width(detections: list[Detection]) -> int:
    if not detections:
        return 1
    return max(det.x2 for det in detections)
