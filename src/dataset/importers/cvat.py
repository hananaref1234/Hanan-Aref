"""Parse CVAT for-images XML (used by the Accessibility Barriers dataset)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class CvatBox:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CvatImage:
    name: str
    width: int
    height: int
    boxes: tuple[CvatBox, ...]
    image_id: str = ""


def parse_cvat_xml(path: Path) -> list[CvatImage]:
    """Parse a CVAT 1.1 annotations.xml file with <image><box> elements."""
    tree = ElementTree.parse(path)
    root = tree.getroot()
    images: list[CvatImage] = []
    for image_el in root.findall("image"):
        boxes: list[CvatBox] = []
        for box_el in image_el.findall("box"):
            attributes = {
                attr.get("name", ""): (attr.text or "").strip()
                for attr in box_el.findall("attribute")
            }
            boxes.append(
                CvatBox(
                    label=str(box_el.get("label", "")),
                    xtl=float(box_el.get("xtl", 0)),
                    ytl=float(box_el.get("ytl", 0)),
                    xbr=float(box_el.get("xbr", 0)),
                    ybr=float(box_el.get("ybr", 0)),
                    attributes=attributes,
                )
            )
        images.append(
            CvatImage(
                name=str(image_el.get("name", "")),
                width=int(image_el.get("width", 0)),
                height=int(image_el.get("height", 0)),
                boxes=tuple(boxes),
                image_id=str(image_el.get("id", "")),
            )
        )
    return images
