# Li-Stick AI — Reproducible training & hardware handoff

## Current source and asset status
- Source code from original project has been imported, preserving `MODELS.md` and `data/sources/datasets.yaml`.
- Source v0.3 checkpoint `models/li_stick_navigation_v0.3_combined.pt` **is missing**. No v0.5 trained model has been produced.
- The image folders in the upstream repository are placeholders. Original labeled images, complete split/provenance CSVs, and external access permissions are **missing**.
- Existing registry reports 13 classes and 9,780 / 533 / 618 train/val/test images (historical counts, not uploaded samples).
- Critical safety labels for potholes, ground drop-offs, and overhead hazards require separate, documented annotation and new model versions.

## Team handoff checklist
1. Provide model `.pt`, SHA256, class map, and exact training config.
2. Provide image/label datasets and immutable manifests or secure storage access. Include all licenses and user consent conditions.
3. Include full original split and provenance CSVs, and record image/video capture groups to prevent train-test leakage.
4. Supply evaluation set representing local indoor/outdoor terrain, curbs, traffic, reflective surfaces, darkness, overhead obstructions, dropped cane events.
5. Supply hardware details: ultrasonic sensor model and min range; IMU part and mount; Grove Vision AI V2 firmware/model compiler; ESP32-S3 board configurations; wiring and calibration.
6. Identify GPU training machine and evaluation owner.

## Host setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -r requirements-integration.txt
python -m unittest discover -s tests -p 'test_next_safety.py' -v
streamlit run ui/dashboard.py
```

## Dataset validation
Edit `config/dataset_v04.yaml` to point to real local images and labels, keeping the 13-class order aligned with the v0.3 model.

```bash
python -m training_v04.audit --data config/dataset_v04.yaml
python -m training_v04.train --data config/dataset_v04.yaml --weights models/li_stick_navigation_v0.3_combined.pt --device 0 --epochs 100
python -m training_v04.evaluate --data config/dataset_v04.yaml --weights outputs/train_v04/experiment/weights/best.pt
python -m training_v04.export --weights outputs/train_v04/experiment/weights/best.pt --imgsz 320
```

These commands are **not** verified training runs: the necessary dataset and checkpoint are missing. The ONNX export is not compatible firmware for Grove Vision AI V2 by itself. Hardware safety remains unvalidated.

## Releasing trained artifacts
Store source/dataset licenses, SHA256 hashes, commit SHA, dataset split manifest, model version, training logs, independently evaluated per-class precision/recall, mAP50 and mAP50:95, p50/p95 alert latency, hardware setup, environmental failures and risk analysis. Never name an untrained candidate as a completed model.
