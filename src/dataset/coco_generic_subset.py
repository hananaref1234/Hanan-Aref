"""Official COCO 2017 filtered subset for selected combined-generic classes.

Downloads annotations and only the sampled JPEGs from images.cocodataset.org.
Does not ingest all 80 COCO classes. Does not fabricate labels.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from src.dataset.importers.sampling import cap_instances_by_image
from src.dataset.taxonomy import COMBINED_DETECTOR_CLASSES
from src.utils.paths import project_root, resolve_from_root

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMAGE_URL = "http://images.cocodataset.org/{split}/{file_name}"
DEFAULT_RAW_DIR = Path("data/raw/coco_2017_generic_subset")
SELECTED_COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
)
DEFAULT_CHAIR_CAP = 2000
DEFAULT_SEED = 42
USER_AGENT = "li-stick-ai/v0.3-combined (official COCO subset download)"


class CocoSubsetError(RuntimeError):
    """Raised when the official COCO subset cannot be prepared."""


@dataclass(frozen=True)
class CocoBox:
    annotation_id: int
    image_id: int
    file_name: str
    coco_split: str
    category_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    image_path: Path


@dataclass(frozen=True)
class CocoSubsetStats:
    raw_dir: Path
    images: int
    annotations: int
    per_class: dict[str, int]
    chair_images: int


def prepare_coco_chair_subset(
    dest: Path | None = None,
    *,
    chair_cap: int = DEFAULT_CHAIR_CAP,
    seed: int = DEFAULT_SEED,
    download: bool = True,
) -> CocoSubsetStats:
    """Create a YOLO-labeled COCO chair-focused subset under ``dest``."""
    dest = resolve_from_root(dest or DEFAULT_RAW_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    annotations_json = _ensure_instances_json(dest, download=download)
    payload = json.loads(annotations_json.read_text(encoding="utf-8"))
    boxes = _parse_selected_boxes(payload, coco_split="train2017")
    chair_grouped: dict[int, list[CocoBox]] = defaultdict(list)
    by_image: dict[int, list[CocoBox]] = defaultdict(list)
    for box in boxes:
        by_image[box.image_id].append(box)
        if box.category_name == "chair":
            chair_grouped[box.image_id].append(box)
    if not chair_grouped:
        raise CocoSubsetError("COCO instances_train2017.json contains no chair boxes.")

    grouped_for_cap = {str(image_id): items for image_id, items in chair_grouped.items()}
    selected_chair = cap_instances_by_image(
        grouped_for_cap,
        max_instances=chair_cap,
        seed=seed,
    )
    selected_ids = sorted({box.image_id for box in selected_chair})
    print(
        f"Sampling {len(selected_ids)} COCO train2017 images "
        f"({sum(len(chair_grouped[i]) for i in selected_ids)} chair boxes)."
    )
    image_dir = dest / "images"
    label_dir = dest / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    written_images = 0
    per_class: dict[str, int] = {name: 0 for name in SELECTED_COCO_CLASSES}
    kept_boxes: list[dict[str, object]] = []
    for image_id in selected_ids:
        image_boxes = by_image[image_id]
        file_name = image_boxes[0].file_name
        dest_name = f"coco_{image_id:012d}{Path(file_name).suffix.lower() or '.jpg'}"
        dest_image = image_dir / dest_name
        if download or not dest_image.exists():
            _download_image(
                COCO_IMAGE_URL.format(split="train2017", file_name=file_name),
                dest_image,
            )
            if written_images and written_images % 50 == 0:
                print(f"Downloaded {written_images}/{len(selected_ids)} COCO images...")
        if not dest_image.exists():
            raise CocoSubsetError(f"Missing COCO image {dest_image}.")
        lines = []
        for box in image_boxes:
            training_id = COMBINED_DETECTOR_CLASSES.index(box.category_name)
            lines.append(
                f"{training_id} {box.x_center:.6f} {box.y_center:.6f} "
                f"{box.width:.6f} {box.height:.6f}"
            )
            per_class[box.category_name] += 1
            kept_boxes.append(
                {
                    "annotation_id": box.annotation_id,
                    "image_id": image_id,
                    "file_name": dest_name,
                    "source_file_name": file_name,
                    "category_name": box.category_name,
                    "coco_split": "train2017",
                }
            )
        (label_dir / f"{Path(dest_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        written_images += 1

    stats = {
        "dataset": "COCO 2017",
        "source": "http://images.cocodataset.org/",
        "license": "annotations CC BY 4.0; images Flickr (per-image licenses)",
        "date_accessed": "2026-08-18",
        "selected_generic_classes": list(SELECTED_COCO_CLASSES),
        "filtering_method": (
            "Official instances_train2017.json; non-crowd boxes; whole-image sample "
            f"of chair images until ~{chair_cap} chair instances (seed {seed}). "
            "Other selected generic boxes on those images are kept."
        ),
        "image_count": written_images,
        "annotation_count": sum(per_class.values()),
        "per_class": per_class,
        "chair_cap": chair_cap,
        "sample_seed": seed,
    }
    (dest / "subset_manifest.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    (dest / "README.md").write_text(
        _readme_text(stats),
        encoding="utf-8",
    )
    return CocoSubsetStats(
        raw_dir=dest,
        images=written_images,
        annotations=int(stats["annotation_count"]),
        per_class=per_class,
        chair_images=written_images,
    )


def load_coco_subset_boxes(raw_dir: Path | None = None) -> list[CocoBox]:
    """Load YOLO boxes written by ``prepare_coco_chair_subset``."""
    raw_dir = resolve_from_root(raw_dir or DEFAULT_RAW_DIR)
    image_dir = raw_dir / "images"
    label_dir = raw_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise CocoSubsetError(f"COCO subset not prepared at {raw_dir}.")
    boxes: list[CocoBox] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        image_id = _image_id_from_dest(image_path.name)
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise CocoSubsetError(f"Malformed COCO subset label {label_path}:{line_number}")
            training_id = int(parts[0])
            name = COMBINED_DETECTOR_CLASSES[training_id]
            boxes.append(
                CocoBox(
                    annotation_id=line_number,
                    image_id=image_id,
                    file_name=image_path.name,
                    coco_split="train2017",
                    category_name=name,
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                    image_path=image_path,
                )
            )
    return boxes


def _ensure_instances_json(dest: Path, *, download: bool) -> Path:
    json_path = dest / "annotations" / "instances_train2017.json"
    if json_path.is_file():
        return json_path
    zip_path = dest / "annotations_trainval2017.zip"
    if download and not zip_path.is_file():
        _download_url(COCO_ANNOTATIONS_URL, zip_path)
    if not zip_path.is_file():
        raise CocoSubsetError(
            f"COCO annotations zip not found at {zip_path}. "
            "Download the official annotations_trainval2017.zip from cocodataset.org."
        )
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith("instances_train2017.json")
        ]
        if not members:
            raise CocoSubsetError("instances_train2017.json missing from COCO annotations zip.")
        archive.extract(members[0], dest)
        extracted = dest / members[0]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if extracted != json_path:
        json_path.write_bytes(extracted.read_bytes())
    return json_path


def _parse_selected_boxes(payload: dict, *, coco_split: str) -> list[CocoBox]:
    categories = {
        int(item["id"]): str(item["name"])
        for item in payload.get("categories", [])
    }
    wanted = {cat_id: name for cat_id, name in categories.items() if name in SELECTED_COCO_CLASSES}
    images = {
        int(item["id"]): item
        for item in payload.get("images", [])
    }
    boxes: list[CocoBox] = []
    for ann in payload.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        category_id = int(ann["category_id"])
        name = wanted.get(category_id)
        if name is None:
            continue
        image = images.get(int(ann["image_id"]))
        if image is None:
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, width, height = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        img_w = float(image["width"])
        img_h = float(image["height"])
        if width <= 0 or height <= 0 or img_w <= 0 or img_h <= 0:
            continue
        boxes.append(
            CocoBox(
                annotation_id=int(ann["id"]),
                image_id=int(image["id"]),
                file_name=str(image["file_name"]),
                coco_split=coco_split,
                category_name=name,
                x_center=(x + width / 2.0) / img_w,
                y_center=(y + height / 2.0) / img_h,
                width=width / img_w,
                height=height / img_h,
                image_path=Path(),
            )
        )
    return boxes


def _download_image(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    _download_url(url, tmp)
    tmp.replace(dest)


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)


def _image_id_from_dest(filename: str) -> int:
    stem = Path(filename).stem
    if stem.startswith("coco_"):
        return int(stem.replace("coco_", "", 1))
    return int(stem)


def _readme_text(stats: dict) -> str:
    per_class = "\n".join(
        f"- {name}: {count}" for name, count in stats["per_class"].items()
    )
    return f"""# COCO 2017 generic subset (Li-Stick v0.3-combined)

Official source: {stats["source"]}

- Dataset: {stats["dataset"]}
- License: {stats["license"]}
- Date accessed: {stats["date_accessed"]}
- Images: {stats["image_count"]}
- Annotations: {stats["annotation_count"]}
- Filtering: {stats["filtering_method"]}

## Selected classes

{per_class}

Do not treat this directory as the full COCO dataset. Custom Li-Stick hazards
are unlabeled here.
"""


def default_raw_dir() -> Path:
    return project_root() / DEFAULT_RAW_DIR
