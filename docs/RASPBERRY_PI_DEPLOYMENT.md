# Raspberry Pi 5 Deployment Guide

Li-Stick Navigation Detector — v0.3-combined  
Target hardware: **Raspberry Pi 5** + **Camera Module V2 NoIR**

---

## Quick Reference

```bash
# 1. Self-check (run first)
python scripts/pi_selfcheck.py

# 2. Live run
python scripts/run_rpi5_live.py

# 3. Benchmark (synthetic frames, no camera needed)
python scripts/pi_benchmark.py

# 4. Full invocation with explicit args
python main.py \
  --config config/config_rpi5_combined.yaml \
  --classes config/classes_v03_combined.yaml
```

---

## 1. Hardware Requirements

| Component | Spec |
|---|---|
| SBC | Raspberry Pi 5 (4 GB or 8 GB RAM recommended) |
| Camera | Camera Module V2 NoIR (IMX219) |
| Storage | microSD ≥ 32 GB (Class 10 / A2) |
| OS | Raspberry Pi OS (Bookworm, 64-bit, Lite or Full) |
| Python | 3.11 (shipped with Bookworm) |
| Power | Official 27 W USB-C PSU |

---

## 2. Final Model

| Field | Value |
|---|---|
| Checkpoint | `models/li_stick_navigation_v0.3_combined.pt` |
| Class map | `models/li_stick_navigation_v0.3_combined.class_map.yaml` |
| Config | `config/config_rpi5_combined.yaml` |
| Classes | `config/classes_v03_combined.yaml` |
| Base model | `yolo26n.pt` (Ultralytics YOLO 2.6 nano) |
| Input size | 640×640 |
| # Classes | 13 |

### Detector I/O Contract

**Input:** `numpy.ndarray` — shape `(H, W, 3)`, dtype `uint8`, BGR colour order  
(Picamera2 backend converts RGB→BGR automatically before handing to `ObjectDetector`.)

**Output:** `list[Detection]`

```
Detection
  .class_id        int     — global Li-Stick class ID (see table below)
  .class_name      str     — human-readable class name
  .confidence      float   — raw YOLO confidence (0–1)
  .bbox            BoundingBox  — (.x1, .y1, .x2, .y2) pixel coords
  .center_x        int     — bbox horizontal midpoint
  .center_y        int     — bbox vertical midpoint
  .direction       Direction   — LEFT / CENTER / RIGHT
  .timestamp       float   — Unix time at frame capture
```

After the tracking pipeline, `TrackedDetection` additionally exposes:

```
TrackedDetection (extends Detection fields)
  .track_id              int
  .smoothed_confidence   float | None
  .stable_direction      Direction | None
  .hit_count             int
  .is_confirmed          bool
```

### Global Class ID Table — v0.3-combined

The model's **training IDs** (0–12, contiguous, required by Ultralytics) are
remapped to **global Li-Stick IDs** by the sidecar class map at inference time.
`ObjectDetector.detect()` always returns global IDs.

| Training ID | Global ID | Class name | Group |
|---|---|---|---|
| 0 | 0 | pole | custom |
| 1 | 1 | bollard | custom |
| 2 | 2 | stairs | custom |
| 3 | 3 | curb | custom |
| 4 | 6 | barrier | custom |
| 5 | 11 | tree_trunk | custom |
| 6 | 12 | person | generic |
| 7 | 13 | bicycle | generic |
| 8 | 14 | car | generic |
| 9 | 15 | motorcycle | generic |
| 10 | 16 | bus | generic |
| 11 | 17 | truck | generic |
| 12 | 18 | chair | generic |

> **Note:** Global IDs 4, 5, 7, 8, 9, 10 correspond to v0.2 custom classes that
> are not included in the v0.3 training subset (open_doorway, closed_door, drop_off,
> pothole, low_overhead_obstacle, wall). They are preserved in the taxonomy for
> future experiments.

---

## 3. Raspberry Pi Setup — Exact Commands

### 3.1 System prerequisites

```bash
sudo apt update && sudo apt upgrade -y

# Camera stack (required for Picamera2)
sudo apt install -y python3-picamera2 python3-libcamera libcamera-apps \
    libcamera-dev python3-prctl

# OpenCV system dependencies
sudo apt install -y libopencv-dev python3-opencv

# Build tools (needed for lap / bytetrack deps)
sudo apt install -y build-essential python3-dev
```

### 3.2 Enable the camera

```bash
sudo raspi-config
# → Interface Options → Camera → Enable
# Reboot:
sudo reboot
```

Verify:
```bash
libcamera-hello --nopreview -t 2000
```

Expected output: camera properties printed, preview frame captured.

### 3.3 Copy the project to the Pi

```bash
# From your development machine:
rsync -avz --exclude='.venv' --exclude='data/' --exclude='runs/' \
  /path/to/li_stick_ai/ pi@<PI_IP>:~/li_stick_ai/

# Only required runtime files need to be on the Pi:
# - main.py
# - src/
# - models/li_stick_navigation_v0.3_combined.pt
# - models/li_stick_navigation_v0.3_combined.class_map.yaml
# - config/config_rpi5_combined.yaml
# - config/classes_v03_combined.yaml
# - scripts/run_rpi5_live.py
# - scripts/pi_selfcheck.py
# - scripts/pi_benchmark.py
# - requirements_rpi5.txt
```

### 3.4 Create virtual environment

```bash
cd ~/li_stick_ai

# --system-site-packages gives the venv access to the apt-installed Picamera2
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
```

### 3.5 Install runtime dependencies

```bash
pip install -r requirements_rpi5.txt
```

> `opencv-python-headless` is used instead of `opencv-python` to avoid pulling in
> Qt/GUI dependencies that are unnecessary for headless Pi operation.

### 3.6 Run the self-check

```bash
python scripts/pi_selfcheck.py
```

All checks should show `[PASS]`. Fix any failures before proceeding.

### 3.7 Test command — single synthetic frame

```bash
python examples/run_detector.py --dummy
```

Expected: `Detections: 0` (black frame → no detections), inference time printed.

### 3.8 Live run command

```bash
python scripts/run_rpi5_live.py
```

Or equivalent full form:

```bash
python main.py \
  --config config/config_rpi5_combined.yaml \
  --classes config/classes_v03_combined.yaml
```

Expected log output (headless, tracking enabled):

```
INFO  Starting Li-Stick Phase 2 detector + tracker
INFO  Model=models/li_stick_navigation_v0.3_combined.pt ...
INFO  [TRACK_NEW] id=3 class=person confidence=0.91 direction=CENTER
INFO  [TRACK_NEW] id=7 class=stairs confidence=0.84 direction=RIGHT
```

Stop with `Ctrl-C`.

### 3.9 Benchmark command

```bash
python scripts/pi_benchmark.py --frames 200
```

Records and prints inference latency, frame latency, FPS, RAM usage.

---

## 4. Performance and Model Optimization Decision Tree

**Do NOT perform any of the following until real Pi performance numbers are measured.**

```
Tomorrow: Run pi_benchmark.py on the actual Pi 5
                    │
                    ▼
        Is achieved FPS acceptable for the application?
        (Suggested minimum: ≥ 5 FPS for navigation aid)
           │                         │
          YES                        NO
           │                         │
    Keep current .pt          Investigate export options:
    No further optimization    1. NCNN export (Ultralytics built-in)
                               2. ONNX + ONNX Runtime
                               3. TFLite (via Ultralytics export)
                               Benchmark each on real Pi hardware,
                               then choose the fastest that keeps
                               mAP50 >= 0.30 on combined test set.
```

**Current model format:** PyTorch `.pt` (not exported)  
**Model optimization status:** NOT YET PERFORMED — pending real Pi benchmark

---

## 5. Pi Camera Test Checklist

Run these manually on the Pi tomorrow:

- [ ] **1. Confirm camera detected**
  ```bash
  libcamera-hello --nopreview -t 2000
  ```
  Expected: camera properties without error.

- [ ] **2. Test Picamera2 Python capture**
  ```python
  from picamera2 import Picamera2
  cam = Picamera2()
  cam.start()
  frame = cam.capture_array("main")
  print(frame.shape)  # expected: (720, 1280, 3)
  cam.stop(); cam.close()
  ```

- [ ] **3. Run detector headless** (over SSH, no monitor)
  ```bash
  python main.py \
    --config config/config_rpi5_combined.yaml \
    --classes config/classes_v03_combined.yaml
  ```
  Expected: inference log lines appear; no GUI window attempted.

- [ ] **4. Test person detection**
  Stand in front of camera. Expect:
  `[TRACK_NEW] id=N class=person confidence=X.XX direction=CENTER`

- [ ] **5. Test stairs detection**
  Point camera at stairs. Expect `class=stairs` events logged.

- [ ] **6. Test pole / barrier detection**
  Point at a lamp post or physical barrier. Expect matching events.

- [ ] **7. Verify LEFT / CENTER / RIGHT**
  Move left-right in front of camera; verify direction changes in logs.

- [ ] **8. Verify tracking IDs**
  Confirm track IDs are stable (same ID for same object across frames)
  and increment correctly when new objects appear.

- [ ] **9. Verify duplicate suppression**
  Keep same object in view for >3 s. Expect only one `TRACK_NEW` event
  per object (not one per frame).

- [ ] **10. Record FPS / latency**
  ```bash
  python scripts/pi_benchmark.py --frames 200
  ```
  Record: inference_ms mean, total frame_ms mean, achieved FPS.

---

## 6. Headless Operation

`config/config_rpi5_combined.yaml` sets `visualization.enabled: false`.

With visualization disabled:
- `cv2.imshow` is **never called**
- `cv2.waitKey` is **never called**
- `cv2.destroyAllWindows` is **never called**
- No display server (X11/Wayland) is required
- Safe to run over SSH with no `DISPLAY` set

Only `opencv-python-headless` is required (no Qt).

---

## 7. Dependency Split

### Runtime (Pi) dependencies — `requirements_rpi5.txt`

| Package | Source | Notes |
|---|---|---|
| `ultralytics` | pip | Inference engine |
| `opencv-python-headless` | pip | Frame I/O, BGR conversion |
| `numpy` | pip | Array handling |
| `PyYAML` | pip | Config loading |
| `lap` | pip | ByteTrack bipartite matching |
| `picamera2` | **apt** | `sudo apt install python3-picamera2` |
| `libcamera` | **apt** | Camera stack |

### Training-only dependencies — NOT installed on Pi

| Package | Reason |
|---|---|
| `roboflow` | Dataset download — not needed at inference |
| `pytest` | Testing only |
| `pycocotools` | Annotation parsing — training only |
| CUDA / GPU drivers | Pi uses CPU inference |
| Notebook packages | Development only |

---

## 8. Known Untested Items

The following require the physical Pi and are scheduled for tomorrow:

- Actual camera capture via Picamera2 (tested only on the import guard level)
- Camera Module V2 NoIR frame format on Pi 5 (RGB888 path assumed correct)
- Real inference latency and FPS on Pi 5 (only laptop numbers exist)
- Thermal throttling under sustained load
- Camera permission setup under a new Pi OS user
- `lap` wheel availability for aarch64 (may need `pip install lap --no-binary :all:` or `python-lap`)

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Picamera2Error: Picamera2 is not installed` | Picamera2 not on venv path | Use `--system-site-packages` venv |
| `libcamera: No cameras available` | Camera not enabled | `sudo raspi-config` → Camera → Enable |
| `DetectorError: Could not load YOLO model` | `.pt` not copied | Copy `models/` directory to Pi |
| `ImportError: cv2` | OpenCV not installed | `pip install opencv-python-headless` |
| `lap` install fails (aarch64) | No prebuilt wheel | `pip install python-lap` or build from source |
| Low FPS (< 2) | CPU bottleneck | See Section 4 optimization decision tree |
| `[FAIL] Picamera2 NOT importable` in selfcheck | apt package missing | `sudo apt install python3-picamera2` |
