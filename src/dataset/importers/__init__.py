"""Source-specific dataset importers."""

from src.dataset.importers.accessibility_barriers import AccessibilityBarriersImporter
from src.dataset.importers.base_importer import BaseImporter, SourceInspectReport
from src.dataset.importers.cityscapes import CityscapesImporter
from src.dataset.importers.doordetect import DoorDetectImporter
from src.dataset.importers.listick_camera import ListickCameraImporter
from src.dataset.importers.mapillary import MapillaryVistasImporter
from src.dataset.importers.open_images import OpenImagesImporter
from src.dataset.importers.pothole import PotholeImporter
from src.dataset.importers.roboflow_yolo import RoboflowYoloImporter
from src.dataset.importers.sideguide import SideGuideImporter
from src.dataset.importers.timbervision import TimberVisionImporter
from src.dataset.manifest import SourceSpec

IMPORTER_CLASSES: dict[str, type[BaseImporter]] = {
    "sideguide": SideGuideImporter,
    "accessibility_barriers": AccessibilityBarriersImporter,
    "doordetect": DoorDetectImporter,
    "cityscapes": CityscapesImporter,
    "open_images_v7": OpenImagesImporter,
    "timbervision": TimberVisionImporter,
    "pothole": PotholeImporter,
    "mapillary_vistas": MapillaryVistasImporter,
    "listick_camera": ListickCameraImporter,
    "roboflow_sidewalk_barriers": RoboflowYoloImporter,
    "roboflow_bollard_v2gn5": RoboflowYoloImporter,
    "roboflow_power_pole": RoboflowYoloImporter,
    "roboflow_bollard_nkhwt": RoboflowYoloImporter,
    "roboflow_asset_model": RoboflowYoloImporter,
    "roboflow_bollards_crosswalk_stairs": RoboflowYoloImporter,
    "roboflow_sideguide_scottsdale": RoboflowYoloImporter,
}


def importer_for(source: SourceSpec, staging_root=None) -> BaseImporter:
    try:
        cls = IMPORTER_CLASSES[source.name]
    except KeyError as exc:
        raise KeyError(f"No importer registered for {source.name!r}.") from exc
    return cls(source, staging_root=staging_root)


__all__ = [
    "IMPORTER_CLASSES",
    "BaseImporter",
    "SourceInspectReport",
    "importer_for",
]
