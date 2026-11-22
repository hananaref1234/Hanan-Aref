# Li-Stick AI — Hanan Aref

This repository contains the **original Li-Stick Python research baseline** from [cezarkhatib/li_stick_ai](https://github.com/cezarkhatib/li_stick_ai) together with ongoing experimental sensor-fusion and training enhancements. Original code resides in `src/`, `scripts/`, `config/` and `main.py`. New experimental modules reside in `li_stick_next/`, `training_v04/`, `ui/`, and `integrated_live.py`.

## Status
**Not a certified navigation or fall-detection device.** The original model registry reports v0.3 combined mAP@50 of 0.331, not new measured accuracy. No trained checkpoint or local images are in this repository. All candidate thresholds require calibration. No Grove Vision AI V2 firmware or on-device model compilation is claimed.

## Set up
```bash
git clone https://github.com/hananaref1234/Hanan-Aref.git
cd Hanan-Aref
python -m venv .venv
pip install -r requirements.txt
pip install -r requirements-integration.txt
python -m unittest discover -s tests -p 'test_next_safety.py' -v
streamlit run ui/dashboard.py
```

To run real YOLO inference, first obtain the properly licensed exact `models/li_stick_navigation_v0.3_combined.pt` checkpoint, and configure input camera and data source. Live hardware use must remain supervised.

## Data and missing assets
See [dataset request](docs/DATASET_REQUEST.md) and [existing dataset source registry](data/sources/datasets.yaml). Historical source files have been imported excluding large provenance/split CSV files and empty dataset placeholders. Image/video data and weights must be supplied by the team or independently acquired under proper licenses. No fabricated weights or benchmark figures have been committed.

## Attribution
Preserve original authorship, dependency terms, third-party dataset licenses, and source provenance when redistributing and preparing any dataset or commercial product. No upstream model performance is a safety certification.
