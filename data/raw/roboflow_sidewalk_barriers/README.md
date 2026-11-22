# Raw Roboflow — Sidewalk Barriers Dataset

**Status (2026-08-18): NOT DOWNLOADED.** Place the official YOLO export here after manual download.

- **Universe URL:** https://universe.roboflow.com/sidewalk-barriers-datasets/sidewalk-barriers
- **License:** CC BY 4.0 — documented on Universe (not APPROVED until human review)
- **Publisher:** Sidewalk Barriers Datasets
- **Scale:** ~24k images
- **Exp2 import classes:** `barricade` (→ barrier review), `bollard`, `tree_trunk`
- **Excluded:** curb, stairs, traffic_sign, etc. (Accessibility Barriers keeps stairs/curb)

## Manual download

1. Open the Universe URL (Roboflow account may be required).
2. **Download Dataset** → **YOLOv8** or **YOLO26** format.
3. Extract so this folder contains `data.yaml`, `train/images`, `train/labels`, etc.
4. Do not edit raw files.
5. Import to staging:

```text
python scripts/import_source.py --source roboflow_sidewalk_barriers --replace-source
python scripts/validate_staging.py
```

Deterministic subsampling caps at **2000 instances/class** (`sample_seed: 42`).
