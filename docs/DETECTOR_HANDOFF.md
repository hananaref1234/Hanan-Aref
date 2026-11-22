# Detector handoff contract (v0.3-combined)

This is the software interface for the Li-Stick navigation detector. It does
**not** include Raspberry Pi, NCNN, ESP32, serial, ultrasonic, IMU, speech,
vibration, depth, or risk-engine logic.

## Input

An OpenCV / NumPy image frame:

- HWC `uint8` array
- BGR (OpenCV default) or RGB
- Typical size 1280×720 from the webcam config; the detector resizes internally to `imgsz=640`

## Output

`list[Detection]` from `ObjectDetector.detect(frame)`.

Each `Detection` exposes at least:

- `class_id` — **global** Li-Stick ID (not the contiguous YOLO training ID)
- `class_name`
- `confidence`
- `bbox` (`x1`, `y1`, `x2`, `y2` in frame pixels)
- `center_x`, `center_y`
- `direction` (`left` / `center` / `right` from image geometry, not stairs up/down)
- `timestamp`

The tracking path can additionally provide `track_id` on tracker outputs.

## Usage

```python
from src.utils.config_loader import load_app_config
from src.vision.detector import ObjectDetector

config = load_app_config(
    "config/config_v03_combined.yaml",
    "config/classes_v03_combined.yaml",
)
detector = ObjectDetector(
    model_path=config.detector.model_path,
    confidence_threshold=config.detector.confidence_threshold,
    allowed_classes=config.allowed_classes,
    image_size=config.detector.image_size,
)
detections, inference_ms = detector.detect(frame)
```

CLI:

```text
python main.py --config config/config_v03_combined.yaml --classes config/classes_v03_combined.yaml
```

## Class remapping

The checkpoint head uses contiguous training IDs. `ObjectDetector` loads
`models/li_stick_navigation_v0.3_combined.class_map.yaml` and emits global IDs:

| Training ID | Name | Global ID |
| --- | --- | --- |
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

Public API is unchanged from earlier experiments: callers still receive `Detection` objects.

## Pipeline

`ObjectDetector` → `Detection` → ByteTrack (`ObjectTracker`) → `TemporalStabilizer` → `EventSuppressor`
