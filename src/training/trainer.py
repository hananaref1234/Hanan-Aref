"""Ultralytics training wrapper. Does not start if the dataset is empty."""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from src.dataset.class_map import load_class_map, sidecar_path_for_checkpoint, write_class_map
from src.dataset.validator import validate_dataset
from src.training.config import TrainingRunConfig, resolve_device

logger = logging.getLogger(__name__)


class TrainingNotReadyError(RuntimeError):
    """Raised when training is requested without a validated labeled dataset."""


def ensure_dataset_ready(dataset_yaml: Path) -> Path:
    """Return the dataset root if it contains labeled images; otherwise raise."""
    dataset_root = dataset_yaml.parent
    report = validate_dataset(dataset_root)
    if report.empty:
        raise TrainingNotReadyError(
            "CUSTOM DATASET REQUIRED\n"
            f"No labeled images found under {dataset_root}.\n"
            "Collect and annotate data before training. "
            "See data/li_stick_navigation/ANNOTATION_GUIDE.md and "
            "docs/DATA_COLLECTION_PLAN.md.\n"
            "TRAINING: NOT STARTED"
        )
    if not report.passed:
        raise TrainingNotReadyError(
            "Dataset validation failed. Fix labels before training.\n"
            f"{report.summary()}"
        )
    return dataset_root


def train_custom_detector(config: TrainingRunConfig, resume: bool = False) -> Path:
    """Run Ultralytics transfer learning. Requires a validated dataset."""
    dataset_root = ensure_dataset_ready(config.dataset_yaml)
    device = resolve_device(config.device)
    last_pt = config.project / config.name / "weights" / "last.pt"
    logger.info(
        "Starting custom detector training base=%s device=%s epochs=%s imgsz=%s batch=%s resume=%s",
        config.base_model,
        device,
        config.epochs,
        config.image_size,
        config.batch_size,
        resume,
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainingNotReadyError(
            "ultralytics is not installed. pip install -r requirements.txt"
        ) from exc

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if resume:
        if not last_pt.exists():
            raise TrainingNotReadyError(
                f"Cannot resume: {last_pt} not found. Start a fresh run without --resume."
            )
        model = YOLO(str(last_pt))
        results = model.train(resume=True, device=device)
    else:
        model = YOLO(config.base_model)
        aug = config.augmentation
        results = model.train(
            data=str(config.dataset_yaml),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            patience=config.patience,
            device=device,
            workers=config.workers,
            project=str(config.project),
            name=config.name,
            pretrained=True,
            exist_ok=True,
            plots=True,
            hsv_h=aug.hsv_h,
            hsv_s=aug.hsv_s,
            hsv_v=aug.hsv_v,
            degrees=aug.degrees,
            translate=aug.translate,
            scale=aug.scale,
            shear=aug.shear,
            perspective=aug.perspective,
            fliplr=aug.fliplr,
            flipud=aug.flipud,
            mosaic=aug.mosaic,
            mixup=aug.mixup,
            erasing=aug.erasing,
            verbose=True,
        )
    elapsed_s = time.perf_counter() - started
    save_dir = Path(getattr(results, "save_dir", config.project / config.name))
    best_pt = save_dir / "weights" / "best.pt"
    registered = None
    if config.register is not None and best_pt.exists():
        registered = _register_checkpoint(best_pt, config.register, config.class_map)
    meta = _training_meta(
        config=config,
        dataset_root=dataset_root,
        device=device,
        started_at=started_at,
        elapsed_s=elapsed_s,
        save_dir=save_dir,
        best_pt=best_pt,
        registered=registered,
    )
    (save_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    logger.info("Training finished device=%s elapsed_s=%.1f save_dir=%s", device, elapsed_s, save_dir)
    return save_dir


def _register_checkpoint(
    best_pt: Path,
    register_path: Path,
    class_map_path: Path | None,
) -> Path:
    register_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, register_path)
    if class_map_path is not None and class_map_path.exists():
        write_class_map(sidecar_path_for_checkpoint(register_path), load_class_map(class_map_path))
    logger.info("Registered checkpoint %s", register_path)
    return register_path


def _training_meta(
    config: TrainingRunConfig,
    dataset_root: Path,
    device: str | int,
    started_at: str,
    elapsed_s: float,
    save_dir: Path,
    best_pt: Path,
    registered: Path | None,
) -> dict[str, object]:
    epochs_completed, early_stopped = _epochs_from_results_csv(save_dir, config.epochs)
    report = validate_dataset(dataset_root)
    return {
        "model_version": config.name,
        "base_model": config.base_model,
        "dataset_yaml": str(config.dataset_yaml),
        "dataset_root": str(dataset_root),
        "device": device if isinstance(device, str) else f"cuda:{device}",
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "patience": config.patience,
        "pretrained": True,
        "epochs_requested": config.epochs,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "started_at_utc": started_at,
        "training_time_seconds": round(elapsed_s, 1),
        "best_checkpoint": str(best_pt) if best_pt.exists() else None,
        "registered_checkpoint": str(registered) if registered else None,
        "class_map": str(config.class_map) if config.class_map else None,
        "augmentation": {
            "hsv_h": config.augmentation.hsv_h,
            "hsv_s": config.augmentation.hsv_s,
            "hsv_v": config.augmentation.hsv_v,
            "degrees": config.augmentation.degrees,
            "translate": config.augmentation.translate,
            "scale": config.augmentation.scale,
            "shear": config.augmentation.shear,
            "perspective": config.augmentation.perspective,
            "fliplr": config.augmentation.fliplr,
            "flipud": config.augmentation.flipud,
            "mosaic": config.augmentation.mosaic,
            "mixup": config.augmentation.mixup,
            "erasing": config.augmentation.erasing,
        },
        "split_image_counts": report.split_image_counts,
        "annotations_total": report.annotations_total,
    }


def _epochs_from_results_csv(save_dir: Path, requested: int) -> tuple[int | None, bool | None]:
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return None, None
    lines = [line for line in csv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0, False
    completed = len(lines) - 1
    return completed, completed < requested
