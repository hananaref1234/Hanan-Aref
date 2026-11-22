"""Opt-in download helpers. Never bypass access controls. Never run from tests."""

from __future__ import annotations

from src.dataset.manifest import SourceSpec

AUTO_DOWNLOAD_BLOCKED = frozenset(
    {
        "sideguide",
        "cityscapes",
        "mapillary_vistas",
        "pothole",
        "doordetect",
        "listick_camera",
        "open_images_v7",
        "timbervision",
        "roboflow_sidewalk_barriers",
        "roboflow_bollard_v2gn5",
        "roboflow_power_pole",
        "roboflow_bollard_nkhwt",
        "roboflow_asset_model",
        "roboflow_bollards_crosswalk_stairs",
        "roboflow_sideguide_scottsdale",
    }
)

PUBLIC_OPT_IN_DOWNLOAD = frozenset({"accessibility_barriers"})


class DownloadBlockedError(RuntimeError):
    """Raised when automatic download is forbidden."""


def acquisition_instructions(source: SourceSpec) -> str:
    """Human-readable steps to obtain a source legitimately."""
    lines = [
        f"Source: {source.name}",
        f"Status: {source.status}",
        f"License: {source.license}",
        f"Official URL: {source.official_url}",
        f"Place files in: {source.raw_dir}",
        f"Disk space: {source.disk_space_hint or 'unknown'}",
        "",
        source.notes.strip(),
        "",
    ]
    if source.name == "sideguide":
        lines.extend(
            [
                "Inspected 2026-08-18: data/raw/sideguide/ has README only (0 images, 0 annotations).",
                "Exp2 training is blocked until the official release is placed there.",
                "",
                "1. Open https://ytaek-oh.github.io/sideguide",
                f"2. Complete the access request form (academic email required): {source.request_url}",
                "   Fields: Name, academic Email, Affiliation, Supervisor, Purpose, optional Homepage, terms agreement.",
                "3. Terms on that form: non-commercial research only; no redistribution of the dataset or derived data.",
                "   If use is commercial, get legal review before requesting. Contact: datahub@aiworkx.ai",
                "4. Wait for the authors to email a download link. Do not scrape, mirror, or bypass AI-Hub.",
                "5. Extract the official archive into data/raw/sideguide/ (prefer the bounding-box set + CSV/JSON labels).",
                "   Published bbox scale: train 249K / val 37K / test 67K images. Keep original files intact.",
                "6. Then run:",
                "   python scripts/inspect_sources.py --source sideguide",
            ]
        )
    elif source.name == "accessibility_barriers":
        lines.extend(
            [
                "License is CC-BY-4.0. Zip is 8.0 GB.",
                f"Download: {source.download_url}",
                "Extract into data/raw/accessibility_barriers/",
                "Expected: images plus CVAT XML annotations.",
                "Optional opt-in (not run by default):",
                "  python scripts/prepare_source_download.py --source accessibility_barriers --download",
            ]
        )
    elif source.name == "doordetect":
        lines.extend(
            [
                "1. Review the GitHub repo (no LICENSE file published).",
                "2. If legal review allows use, download from Google Drive:",
                f"   {source.download_url}",
                "3. Extract so that data/raw/doordetect/ contains the Door Classification",
                "   and/or Door Detection folders from the README.",
                "4. Cite Ramôa, Alexandre, Mogo, ICARSC 2020.",
            ]
        )
    elif source.name == "cityscapes":
        lines.extend(
            [
                "1. Read https://www.cityscapes-dataset.com/license/",
                "2. Register and download leftImg8bit_trainvaltest and gtFine_trainvaltest",
                "   only after legal review (non-commercial, no redistribution).",
                "3. Extract to data/raw/cityscapes/leftImg8bit and data/raw/cityscapes/gtFine",
                "4. Do not map vegetation to tree_trunk.",
            ]
        )
    elif source.name == "pothole":
        lines.extend(
            [
                "No dataset is selected.",
                "Choose one source with a documented license (CC BY 4.0 preferred).",
                "Fill official_url, author, license, image count, and format in",
                "data/sources/datasets.yaml before importing.",
                "Do not grab a random Roboflow project by keyword.",
            ]
        )
    elif source.name == "mapillary_vistas":
        lines.extend(
            [
                "DOWNLOAD DISABLED.",
                "Status is LICENSE_REVIEW_REQUIRED. Do not download or integrate.",
            ]
        )
    elif source.name == "listick_camera":
        lines.extend(
            [
                "Capture cane-mounted Camera Module V2 NoIR footage locally.",
                "Store originals in data/raw/listick_camera/ with session/video IDs.",
            ]
        )
    elif source.name.startswith("roboflow_"):
        lines.extend(_roboflow_instructions(source))
    elif source.name == "open_images_v7":
        lines.extend(
            [
                "Open Images V7 has no Li-Stick Exp2 bbox classes (pole, bollard, barrier, tree_trunk).",
                "Recorded for research only. Do not map Tree or Street light automatically.",
                "Official site: https://storage.googleapis.com/openimages/web/index.html",
                "Annotations: CC BY 4.0. Images: CC BY 2.0 (verify per-image if required).",
            ]
        )
    elif source.name == "timbervision":
        lines.extend(
            [
                "LICENSE_REVIEW_REQUIRED. CC BY-NC-SA 4.0 — non-commercial, share-alike.",
                "Forestry-domain trunks/logs; not sidewalk tree_trunk semantics.",
                "Official: https://zenodo.org/records/14825846",
                "GitHub: https://github.com/timbervision/timbervision",
                "Do not download until legal review approves NC-SA terms for Li-Stick.",
            ]
        )
    return "\n".join(lines) + "\n"


def _roboflow_instructions(source: SourceSpec) -> list[str]:
    return [
        f"Roboflow Universe page: {source.official_url}",
        f"License: {source.license} ({source.license_url})",
        "Manual download (Roboflow account may be required):",
        "  1. Open the Universe URL above.",
        "  2. Click Download Dataset → choose YOLOv8 (or YOLO26) format.",
        "  3. Extract so images/ and labels/ (plus data.yaml) sit under:",
        f"     {source.raw_dir}",
        "  4. Do not modify raw files. Then run:",
        f"     python scripts/import_source.py --source {source.name} --replace-source",
        "",
        "Automatic Universe download is disabled. Do not scrape or bypass login.",
    ]


def assert_download_allowed(source: SourceSpec) -> None:
    if source.name in AUTO_DOWNLOAD_BLOCKED or source.name not in PUBLIC_OPT_IN_DOWNLOAD:
        raise DownloadBlockedError(
            f"Automatic download of {source.name!r} is disabled "
            f"(status={source.status}, method={source.download_method})."
        )
    if source.status in {"LICENSE_REVIEW_REQUIRED", "REJECTED"}:
        raise DownloadBlockedError(
            f"Refusing to download {source.name!r} with status {source.status}."
        )
