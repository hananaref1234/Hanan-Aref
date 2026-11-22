"""YOLO label parsing and geometric checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

NORMALIZED_TOLERANCE = 0.02


class LabelError(ValueError):
    """Raised when a YOLO annotation line or file is invalid."""


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line_number: int


@dataclass(frozen=True)
class YoloPolygon:
    class_id: int
    points: tuple[tuple[float, float], ...]
    line_number: int


@dataclass(frozen=True)
class YoloAnnotation:
    class_id: int
    line_number: int
    source_format: Literal["BBOX", "POLYGON"]
    x_center: float
    y_center: float
    width: float
    height: float
    polygon_points: tuple[tuple[float, float], ...] = ()

    def to_box(self) -> YoloBox:
        return YoloBox(
            class_id=self.class_id,
            x_center=self.x_center,
            y_center=self.y_center,
            width=self.width,
            height=self.height,
            line_number=self.line_number,
        )


def parse_yolo_label_file(
    path: Path,
    max_class_id: int,
) -> list[YoloBox]:
    """Parse a YOLO txt label file.

    Empty files are valid (hard-negative images) and return no boxes.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelError(f"Could not read label file {path}: {exc}") from exc

    boxes: list[YoloBox] = []
    for annotation in parse_yolo_annotation_text(text, max_class_id, path):
        if annotation.source_format != "BBOX":
            location = f"{path}:{annotation.line_number}"
            raise LabelError(
                f"Polygon YOLO annotation found at {location}, but this caller "
                "expects 5-field detection boxes only."
            )
        boxes.append(annotation.to_box())
    return boxes


def parse_yolo_line(
    line: str,
    line_number: int,
    max_class_id: int,
    path: Path | None = None,
) -> YoloBox:
    """Parse one `class x y w h` YOLO line with normalized coordinates."""
    annotation = parse_yolo_annotation_line(line, line_number, max_class_id, path)
    if annotation.source_format != "BBOX":
        location = f"{path}:{line_number}" if path is not None else f"line {line_number}"
        raise LabelError(
            f"Malformed YOLO label at {location}: expected 5 fields "
            "(class_id x_center y_center width height), got polygon coordinates."
        )
    return annotation.to_box()


def parse_yolo_annotation_file(
    path: Path,
    max_class_id: int,
) -> list[YoloAnnotation]:
    """Parse a YOLO txt file containing bbox and/or polygon annotations."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelError(f"Could not read label file {path}: {exc}") from exc
    return parse_yolo_annotation_text(text, max_class_id, path)


def parse_yolo_annotation_text(
    text: str,
    max_class_id: int,
    path: Path | None = None,
) -> list[YoloAnnotation]:
    annotations: list[YoloAnnotation] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        annotations.append(parse_yolo_annotation_line(line, line_number, max_class_id, path))
    return annotations


def parse_yolo_annotation_line(
    line: str,
    line_number: int,
    max_class_id: int,
    path: Path | None = None,
) -> YoloAnnotation:
    """Parse one YOLO bbox or polygon line with normalized coordinates."""
    parts = line.split()
    location = f"{path}:{line_number}" if path is not None else f"line {line_number}"
    if len(parts) < 5:
        raise LabelError(
            f"Malformed YOLO label at {location}: expected 5 fields minimum, "
            f"got {len(parts)}: {line!r}."
        )

    class_id = _parse_class_id(parts[0], max_class_id, location)
    values = _parse_numeric_values(parts[1:], location, line)

    if len(parts) == 5:
        x_center, y_center, width, height = (
            _clamp_normalized(values[0], "x_center", location),
            _clamp_normalized(values[1], "y_center", location),
            _clamp_normalized(values[2], "width", location),
            _clamp_normalized(values[3], "height", location),
        )
        if width <= 0.0 or height <= 0.0:
            raise LabelError(
                f"Box width/height must be > 0 at {location}, got width={width} height={height}."
            )
        return YoloAnnotation(
            class_id=class_id,
            line_number=line_number,
            source_format="BBOX",
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        )

    coord_count = len(parts) - 1
    if coord_count % 2 != 0:
        raise LabelError(
            f"Malformed polygon at {location}: expected an even number of coordinates "
            f"after class_id, got {coord_count}."
        )
    if coord_count < 6:
        raise LabelError(
            f"Malformed polygon at {location}: need at least 3 points (6 coordinates), "
            f"got {coord_count}."
        )
    points: list[tuple[float, float]] = []
    for index in range(0, len(values), 2):
        x = _clamp_normalized(values[index], f"x{index // 2 + 1}", location)
        y = _clamp_normalized(values[index + 1], f"y{index // 2 + 1}", location)
        points.append((x, y))
    polygon = YoloPolygon(class_id=class_id, points=tuple(points), line_number=line_number)
    return _polygon_to_annotation(polygon)


def parse_yolo_polygon_line(
    line: str,
    line_number: int,
    max_class_id: int,
    path: Path | None = None,
) -> YoloAnnotation:
    """Parse one polygon-only YOLO line.

    Use this in tests or importer-specific code when the caller already knows the
    source line should be polygon data rather than 5-field detection format.
    """
    parts = line.split()
    location = f"{path}:{line_number}" if path is not None else f"line {line_number}"
    if len(parts) == 5:
        raise LabelError(
            f"Malformed polygon at {location}: need at least 3 points (6 coordinates), got 4."
        )
    annotation = parse_yolo_annotation_line(line, line_number, max_class_id, path)
    if annotation.source_format != "POLYGON":
        raise LabelError(f"Expected polygon annotation at {location}.")
    return annotation


def _polygon_to_annotation(polygon: YoloPolygon) -> YoloAnnotation:
    xs = [point[0] for point in polygon.points]
    ys = [point[1] for point in polygon.points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0.0 or height <= 0.0:
        raise LabelError(
            f"Polygon at line {polygon.line_number} has zero area after bounding-box conversion."
        )
    return YoloAnnotation(
        class_id=polygon.class_id,
        line_number=polygon.line_number,
        source_format="POLYGON",
        x_center=(x_min + x_max) / 2.0,
        y_center=(y_min + y_max) / 2.0,
        width=width,
        height=height,
        polygon_points=polygon.points,
    )


def _parse_class_id(raw: str, max_class_id: int, location: str) -> int:
    try:
        class_id = int(raw)
    except ValueError as exc:
        raise LabelError(f"Malformed YOLO label at {location}: class id {raw!r} is not an int.") from exc
    if class_id < 0 or class_id > max_class_id:
        raise LabelError(
            f"Unknown class id {class_id} at {location}. Valid ids are 0-{max_class_id}."
        )
    return class_id


def _parse_numeric_values(parts: list[str], location: str, line: str) -> list[float]:
    try:
        values = [float(item) for item in parts]
    except ValueError as exc:
        raise LabelError(
            f"Malformed YOLO label at {location}: non-numeric values in {line!r}."
        ) from exc
    for value in values:
        if value != value or value in {float("inf"), float("-inf")}:
            raise LabelError(f"Malformed YOLO label at {location}: non-finite coordinate {value}.")
    return values


def _clamp_normalized(value: float, name: str, location: str) -> float:
    if value < -NORMALIZED_TOLERANCE or value > 1.0 + NORMALIZED_TOLERANCE:
        raise LabelError(
            f"{name}={value} at {location} is outside the allowed normalized range "
            f"[{-NORMALIZED_TOLERANCE}, {1.0 + NORMALIZED_TOLERANCE}]."
        )
    return min(max(value, 0.0), 1.0)
