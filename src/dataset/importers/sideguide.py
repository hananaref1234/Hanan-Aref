"""SideGuide importer. Converts only after a legitimate download is placed on disk."""

from __future__ import annotations

import json
from pathlib import Path

from src.dataset.importers.base_importer import (
    BaseImporter,
    ParsedObject,
    SourceInspectReport,
    raw_has_payload,
)
from src.dataset.importers.boxes import PixelBox
from src.dataset.importers.staging import IMAGE_SUFFIXES, infer_sequence_id


class SideGuideImporter(BaseImporter):
    source_name = "sideguide"

    def inspect(self) -> SourceInspectReport:
        raw = self.source.raw_path
        present = raw_has_payload(raw)
        json_files = list(raw.rglob("*.json")) if raw.is_dir() else []
        images = (
            [path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
            if raw.is_dir()
            else []
        )
        detected = ""
        if json_files:
            detected = "json"
        return SourceInspectReport(
            source_name=self.source.name,
            raw_dir=raw,
            present=present,
            annotation_files=len(json_files),
            image_files=len(images),
            detected_format=detected,
            notes=(
                "SideGuide requires access approval. "
                "Place the official release under data/raw/sideguide/."
                if not present
                else "Raw payload found."
            ),
        )

    def parse_objects(self) -> list[ParsedObject]:
        raw = self.source.raw_path
        objects: list[ParsedObject] = []
        for json_path in sorted(raw.rglob("*.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            objects.extend(self._parse_payload(json_path, payload, raw))
        return objects

    def _parse_payload(
        self,
        json_path: Path,
        payload: object,
        raw: Path,
    ) -> list[ParsedObject]:
        if isinstance(payload, dict) and "images" in payload and "annotations" in payload:
            return self._parse_coco(payload, raw)
        if isinstance(payload, dict) and "images" in payload:
            return self._parse_simple(payload, raw)
        if isinstance(payload, list):
            return self._parse_simple({"images": payload}, raw)
        raise ValueError(f"Unrecognized SideGuide JSON schema: {json_path}")

    def _parse_simple(self, payload: dict, raw: Path) -> list[ParsedObject]:
        objects: list[ParsedObject] = []
        for image in payload.get("images", []):
            rel = str(image.get("file_name") or image.get("path") or "")
            image_path = _resolve_image(raw, rel)
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            split = str(image.get("split") or _split_from_path(image_path, raw))
            for box in image.get("boxes", []):
                objects.append(
                    ParsedObject(
                        image_path=image_path,
                        source_image=rel or str(image_path.relative_to(raw)),
                        source_split=split,
                        sequence_id=str(image.get("sequence_id") or infer_sequence_id(image_path)),
                        original_class=str(box.get("class") or box.get("category") or ""),
                        box=PixelBox(
                            float(box["xtl"] if "xtl" in box else box["x1"]),
                            float(box["ytl"] if "ytl" in box else box["y1"]),
                            float(box["xbr"] if "xbr" in box else box["x2"]),
                            float(box["ybr"] if "ybr" in box else box["y2"]),
                        ),
                        image_width=width,
                        image_height=height,
                        conversion_method="bbox",
                    )
                )
        return objects

    def _parse_coco(self, payload: dict, raw: Path) -> list[ParsedObject]:
        categories = {
            int(item["id"]): str(item["name"])
            for item in payload.get("categories", [])
        }
        images = {int(item["id"]): item for item in payload.get("images", [])}
        objects: list[ParsedObject] = []
        for annotation in payload.get("annotations", []):
            image_info = images[int(annotation["image_id"])]
            rel = str(image_info.get("file_name") or "")
            image_path = _resolve_image(raw, rel)
            x, y, width, height = annotation["bbox"]
            objects.append(
                ParsedObject(
                    image_path=image_path,
                    source_image=rel,
                    source_split=str(image_info.get("split") or _split_from_path(image_path, raw)),
                    sequence_id=infer_sequence_id(image_path),
                    original_class=categories.get(int(annotation["category_id"]), ""),
                    box=PixelBox(float(x), float(y), float(x + width), float(y + height)),
                    image_width=int(image_info["width"]),
                    image_height=int(image_info["height"]),
                    conversion_method="bbox",
                    original_annotation=f"coco_ann={annotation.get('id')}",
                )
            )
        return objects


def _resolve_image(raw: Path, rel: str) -> Path:
    candidate = Path(rel)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = raw / rel
    if direct.exists():
        return direct
    matches = list(raw.rglob(Path(rel).name))
    if matches:
        return matches[0]
    return direct


def _split_from_path(image_path: Path, raw: Path) -> str:
    try:
        parts = {part.lower() for part in image_path.relative_to(raw).parts}
    except ValueError:
        parts = {part.lower() for part in image_path.parts}
    for name in ("train", "val", "valid", "validation", "test"):
        if name in parts:
            return "val" if name in {"valid", "validation"} else name
    return ""
