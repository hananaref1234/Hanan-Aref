# Li-Stick AI

Computer-vision navigation for a smart assistive cane.

This repository contains a **local** object-detection and tracking pipeline, a custom navigation-class dataset pipeline, trained YOLO checkpoints, and Raspberry Pi 5 deployment scaffolding. It is a **prototype**, not a certified safety system.

A single detection is never guaranteed truth. Confidence scores stay visible, thresholds are configurable, and this code does **not** invent distance from bounding-box size.

---

## Current status (summary)

| Area | Status |
|---|---|
| Live detection + tracking pipeline | Done |
| Taxonomy v0.2 (12 custom classes) | Frozen |
| Custom-only detectors (Exp1 / Exp2 / Exp2b) | Trained; Exp2b is the custom baseline |
| **Combined detector v0.3** (6 hazards + 7 generics) | **Current production candidate** |
| Raspberry Pi 5 config + headless runtime | Ready (not yet measured on real Pi hardware) |
| ESP32 / serial / speech / vibration / depth / risk | Not implemented |
| NCNN / ONNX / quantization | Not done — wait for real Pi benchmarks |

**Current detector:** `models/li_stick_navigation_v0.3_combined.pt`  
**Class map:** `models/li_stick_navigation_v0.3_combined.class_map.yaml`  
**Desktop config:** `config/config_v03_combined.yaml` + `config/classes_v03_combined.yaml`  
**Pi config:** `config/config_rpi5_combined.yaml`

Full checkpoint registry: [`MODELS.md`](MODELS.md)

> **Git note:** `*.pt` is listed in `.gitignore`, so a normal `git push` will **not** upload weight files. Push the code + docs as usual, then copy or release the `.pt` checkpoints separately (or force-add them if you intentionally want them in the remote). Class-map YAML files are not ignored and will be included.

---

## What we built

### 1. Runtime pipeline (Phases 1–2)

```text
Camera (OpenCV webcam / video file / Picamera2)
  → ObjectDetector (Ultralytics YOLO)
  → Detection objects
  → ByteTrack
  → TemporalStabilizer
  → EventSuppressor
  → logging / optional visualization
```

**Phase 1 — detection**

- Lightweight YOLO nano inference (`yolo26n.pt` base; custom checkpoints replace weights only)
- Confidence filtering and optional class filtering
- Bounding boxes in original-frame pixel coordinates
- LEFT / CENTER / RIGHT from box center (image geometry — not stairs up/down)
- OpenCV camera / video; Picamera2 for Raspberry Pi Camera Module V2 NoIR

**Phase 2 — tracking and events**

- Persistent `track_id` across frames (ByteTrack, wrapped so the app never depends on Ultralytics tracker types)
- Confirmation frames before `TRACK_NEW`
- Missed-frame tolerance before `TRACK_LOST`
- EMA-smoothed confidence + majority-vote stable direction
- Duplicate suppression: one `TRACK_NEW`, silence while unchanged, `TRACK_UPDATE` on real direction change, one `TRACK_LOST`

Only `src/vision/detector.py` talks to YOLO. Only `src/tracking/tracker.py` talks to ByteTrack. The rest of the app consumes `Detection` / `TrackedDetection`.

### 2. Dataset and taxonomy (Phase 3)

**Taxonomy v0.2** freezes 12 custom navigation classes (IDs 0–11):

`pole`, `bollard`, `stairs`, `curb`, `open_doorway`, `closed_door`, `barrier`, `drop_off`, `pothole`, `low_overhead_obstacle`, `wall`, `tree_trunk`

Direction for stairs/curb is **metadata**, not a separate detector class. Bounding-box-only up/down is not treated as reliable (see [`docs/PERCEPTION_FEASIBILITY.md`](docs/PERCEPTION_FEASIBILITY.md)).

Infrastructure includes:

- Source/license manifest (`data/sources/datasets.yaml`)
- Importers, staging, review workflow
- Dataset validator, leakage checks, stratified splits
- Contiguous training-ID ↔ global-ID class maps (sidecar `.class_map.yaml`)
- Transfer-learning trainer and evaluator

### 3. Detector experiments

Earlier runs trained **custom-only** 6-class heads (pole, bollard, stairs, curb, barrier, tree_trunk):

| Experiment | Checkpoint | Role |
|---|---|---|
| Exp1 | `li_stick_navigation_v0.1.pt` | First custom prototype |
| Exp2 | `li_stick_navigation_v0.2.pt` | Expanded custom data |
| Exp2b | `li_stick_navigation_v0.2_exp2b.pt` | Canonical custom-only baseline |

**Design constraint for production:** run **one** detector, not two models (custom + COCO) in parallel.

### 4. Combined detector — `li-stick-navigation-v0.3-combined`

One YOLO head with **13 classes**:

| Group | Classes |
|---|---|
| Custom hazards | pole, bollard, stairs, curb, barrier, tree_trunk |
| Generics | person, bicycle, car, motorcycle, bus, truck, chair |

**Why combined**

- Fine-tuning only custom classes would replace the pretrained COCO head; generic outputs are not assumed to survive custom-only training.
- A single model keeps Pi latency and software integration simpler.

**Data approach**

- Custom boxes: approved Exp2b canonical data only (no pending pole annotations)
- Generics: Roboflow Sidewalk Barriers labels where available + filtered COCO 2017 subset (especially **chair**, absent from Sidewalk Barriers)
- Derived training view: `data/training/li_stick_v03_combined/` (canonical data untouched)
- Group-aware stratified split (~80/10/10) with leakage checks
- Partial-label risk documented (e.g. unlabeled people on custom-only images; unlabeled hazards on COCO images)

**Training**

- Base: `yolo26n.pt` (transfer learning, not from scratch)
- `imgsz=640`, `batch=16`, max 100 epochs, patience 20
- Early stopped at epoch **92** (~20.9 h on RTX 4060 Laptop)
- Contiguous YOLO training IDs remapped to global Li-Stick IDs via sidecar class map

**Held-out test (overall):** mAP50 **0.331**, mAP50-95 **0.215**, P **0.664**, R **0.365**

| Stronger classes | Weaker classes |
|---|---|
| barrier, stairs, bicycle, person, car, bus | pole, tree_trunk, motorcycle, truck, chair |

Comparisons vs Exp2b (custom) and pretrained `yolo26n` (generics): `outputs/evaluation_v03_combined/`

### 5. Raspberry Pi 5 deployment prep

Deployment scaffolding is in place; **physical Pi benchmarking is still pending**.

- Pi config: `config/config_rpi5_combined.yaml` (Picamera2, 1280×720@30, visualization off)
- Runtime deps only: `requirements_rpi5.txt` (no Roboflow / training / CUDA)
- Self-check: `python scripts/pi_selfcheck.py`
- Live: `python scripts/run_rpi5_live.py`
- Benchmark (synthetic frames until real camera): `python scripts/pi_benchmark.py`
- Guide: [`docs/RASPBERRY_PI_DEPLOYMENT.md`](docs/RASPBERRY_PI_DEPLOYMENT.md)

**Optimization policy:** measure `.pt` on the real Pi first. Export (NCNN/ONNX/TFLite) only if FPS is too low.

---

## Detector I/O contract

**Input:** NumPy `uint8` HWC image (BGR from OpenCV / Picamera2 backend).

**Output:** `list[Detection]` with at least:

`class_id`, `class_name`, `confidence`, `bbox`, `center_x`, `center_y`, `direction`, `timestamp`

After tracking: `track_id` (and smoothed confidence / stable direction).

Global IDs for the 13 trained classes (training ID → global):

| Train | Name | Global |
|---|---|---|
| 0 | pole | 0 |
| 1 | bollard | 1 |
| 2 | stairs | 2 |
| 3 | curb | 3 |
| 4 | barrier | 6 |
| 5 | tree_trunk | 11 |
| 6 | person | 12 |
| 7 | bicycle | 13 |
| 8 | car | 14 |
| 9 | motorcycle | 15 |
| 10 | bus | 16 |
| 11 | truck | 17 |
| 12 | chair | 18 |

Details: [`docs/DETECTOR_HANDOFF.md`](docs/DETECTOR_HANDOFF.md)

---

## Explicitly not implemented

- Stairs / curb **up vs down** as a reliable detector output
- Sensor fusion (ultrasonic, IMU)
- Serial / ESP32 firmware
- Speech / vibration alerts
- Hazard risk prioritization or motion commands
- Cloud APIs / LLMs (all detection is local)
- Model export / quantization (pending real Pi numbers)

---

## Setup

Python **3.11+** (Windows development; Raspberry Pi OS Bookworm for deployment).

### Desktop / laptop

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Raspberry Pi (runtime only)

See [`docs/RASPBERRY_PI_DEPLOYMENT.md`](docs/RASPBERRY_PI_DEPLOYMENT.md). Short version:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements_rpi5.txt
# Picamera2 via apt, not pip:
sudo apt install python3-picamera2 python3-libcamera libcamera-apps
```

---

## Usage

Run from the repository root with the venv active.

### Combined detector — webcam (desktop)

```bash
python main.py \
  --config config/config_v03_combined.yaml \
  --classes config/classes_v03_combined.yaml
```

### Combined detector — Raspberry Pi (headless)

```bash
python scripts/pi_selfcheck.py
python scripts/run_rpi5_live.py
```

### Minimal inference example

```bash
python examples/run_detector.py --image path/to/image.jpg
python examples/run_detector.py --dummy
```

### Still image / video helpers

```bash
python scripts/test_image.py path/to/image.jpg
python scripts/test_video.py sample.mp4 --no-display
```

### Training / evaluation (already done for v0.3; do not overwrite casually)

```bash
python scripts/prepare_training_view_v03_combined.py
python scripts/train_custom_detector.py --config config/training_v03_combined.yaml
python scripts/evaluate_v03_combined.py
```

### Tests

```bash
pytest
```

---

## Configuration

| File | Purpose |
|---|---|
| `config/config_v03_combined.yaml` | Desktop webcam + v0.3-combined model |
| `config/classes_v03_combined.yaml` | 13-class filter |
| `config/config_rpi5_combined.yaml` | Pi Picamera2, headless |
| `config/training_v03_combined.yaml` | Training hyperparameters |
| `config/config.yaml` | Legacy / default COCO demo config |

Tunable at runtime: camera size/FPS, `image_size`, confidence, class filter, visualization on/off, tracking on/off.

---

## Project layout

```text
li_stick_ai/
├── main.py                      Live loop (detect + track + events)
├── MODELS.md                    Checkpoint registry
├── requirements.txt             Desktop / training-friendly deps
├── requirements_rpi5.txt        Pi inference-only deps
├── config/                      Runtime + training YAML
├── models/                      .pt checkpoints + .class_map.yaml sidecars
├── examples/run_detector.py     Minimal ObjectDetector demo
├── scripts/
│   ├── run_rpi5_live.py         Pi entrypoint
│   ├── pi_selfcheck.py          Environment checks
│   ├── pi_benchmark.py          Latency / FPS benchmark
│   ├── train_custom_detector.py
│   └── evaluate_v03_combined.py
├── src/
│   ├── camera/                  OpenCV + Picamera2
│   ├── vision/                  ObjectDetector, Detection, direction
│   ├── tracking/                ByteTrack, stabilizer, events
│   ├── dataset/                 Taxonomy, class maps, combined view, validators
│   ├── training/                Trainer, evaluator, comparisons
│   ├── visualization/           Optional overlays
│   └── utils/                   Config, paths, logging
├── data/
│   ├── sources/                 Dataset licenses / mappings
│   ├── raw/                     Third-party downloads
│   ├── staging/                 Review staging
│   ├── li_stick_navigation/     Canonical custom dataset
│   └── training/                Derived YOLO views (v02_exp2b, v03_combined, …)
├── docs/                        Taxonomy, strategy, Pi deployment, handoff
├── outputs/                     Evaluation reports
└── tests/                       pytest (no camera required)
```

---

## Documentation index

| Doc | Contents |
|---|---|
| [`MODELS.md`](MODELS.md) | All trained checkpoints and metrics |
| [`docs/TAXONOMY.md`](docs/TAXONOMY.md) | Class IDs and versioning rules |
| [`docs/CLASS_STRATEGY.md`](docs/CLASS_STRATEGY.md) | Custom vs generic strategy |
| [`docs/DETECTOR_HANDOFF.md`](docs/DETECTOR_HANDOFF.md) | Software I/O contract |
| [`docs/RASPBERRY_PI_DEPLOYMENT.md`](docs/RASPBERRY_PI_DEPLOYMENT.md) | Pi install, checklist, optimization tree |
| [`docs/DATASET_ACQUISITION.md`](docs/DATASET_ACQUISITION.md) | Source acquisition |
| [`docs/PERCEPTION_FEASIBILITY.md`](docs/PERCEPTION_FEASIBILITY.md) | What bbox-only can / cannot do |
| [`docs/DOMAIN_GAP_PLAN.md`](docs/DOMAIN_GAP_PLAN.md) | Domain-gap mitigation |

---

## Safety notes

- Do not treat a detection or track as a physical guarantee.
- Do not infer distance from box size. `Detection.depth_m` stays unset until a real depth source exists.
- This prototype does not command motion, speech, or vibration.
- ESP32 / sensor fusion is planned later; it is not part of the current runtime.
