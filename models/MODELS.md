# Model registry

Weights are gitignored (`*.pt`). This file is the traceability record.

| Field | v0 (current production) | v0.2-exp1 | v0.2-exp2 |
| --- | --- | --- | --- |
| Model version | yolo26n pretrained COCO | `li-stick-navigation-v0.2-exp1` | `li-stick-navigation-v0.2-exp2` |
| Checkpoint | `yolo26n.pt` | `models/li_stick_navigation_v0.2_exp1.pt` | `models/li_stick_navigation_v0.2_exp2.pt` |
| Sidecar class map | none | `models/li_stick_navigation_v0.2_exp1.class_map.yaml` | `models/li_stick_navigation_v0.2_exp2.class_map.yaml` |
| Base checkpoint | yolo26n.pt | yolo26n.pt (transfer learning) | yolo26n.pt (transfer learning) |
| Dataset version | COCO (Ultralytics) | `data/training/li_stick_v02_exp1/` | `data/training/li_stick_v02_exp2/` |
| Taxonomy | COCO 80 | Training 0=stairs→2, 1=curb→3 | Training 0=pole→0, 1=bollard→1, 2=stairs→2, 3=curb→3, 4=barrier→6, 5=tree_trunk→11 |
| Trained classes | COCO 80, filtered in `config/classes.yaml` | stairs, curb | pole, bollard, stairs, curb, barrier, tree_trunk |
| Training date | n/a | 2026-08-17 → 2026-08-18 | 2026-08-18 |
| Device | n/a | CUDA:0 RTX 4060 Laptop GPU (8 GB) | CUDA:0 RTX 4060 Laptop GPU (8 GB) |
| Image size / batch / epochs | n/a | 640 / 16 / 80 requested, 80 completed, early stop no | 640 / 16 / 100 requested, 93 completed, early stop **yes** (patience 20) |
| Active training time | n/a | see run dir | 4054.9 s (~67.6 min) |
| Held-out **test** metrics | not a Li-Stick eval | P=0.623 R=0.478 mAP50=0.424 mAP50-95=0.271 | P=0.531 R=0.461 mAP50=0.417 mAP50-95=0.275 |
| stairs (test) | n/a | P=0.770 R=0.664 mAP50=0.635 mAP50-95=0.419 | P=0.797 R=0.637 mAP50=0.617 mAP50-95=0.430 |
| curb (test) | n/a | P=0.476 R=0.291 mAP50=0.212 mAP50-95=0.124 | P=0.647 R=0.320 mAP50=0.273 mAP50-95=0.166 |
| Export format | PyTorch `.pt` | PyTorch `.pt` (no NCNN yet) | PyTorch `.pt` (no NCNN yet) |

## Generic COCO preservation

**NOT YET IMPLEMENTED.** Exp2 and Exp2b are custom-hazard experiments only. Those checkpoints do **not** preserve or output person / bicycle / car / motorcycle / bus / truck / chair (or any other COCO class). A later combined generic+custom experiment is required. Do not run two permanent detectors as a workaround.

## v0.2-exp1 known limitations

- Experimental only. Not a safety-certified model. Not fall-prevention. Not navigation-ready.
- No up/down / direction classification.
- No training data for pole, bollard, doors, barrier, drop_off, pothole, low_overhead_obstacle, wall, tree_trunk.
- Images are public Accessibility Barriers stills, not cane-camera / Pi Camera Module V2.
- Desktop GPU timings must not be treated as Raspberry Pi 5 performance.
- Curb detection is weak (low recall). Stairs/curb almost never confused with each other; misses and extra boxes dominate.

## v0.2-exp2 known limitations

- Experimental only. Not a safety-certified model. Not fall-prevention. Not navigation-ready.
- Custom hazard classes only. **Does not emit COCO/generic classes.**
- 1,418 pole rows remain PENDING and were not used.
- Source-defined Roboflow splits were preserved: val has **0 pole and 0 bollard**; test has only **4 bollard** and **7 tree_trunk**. Bollard test metrics (0.0) are not statistically meaningful.
- Pole recall is low (0.165). Many thin distant poles in cluttered street scenes are missed.
- Images are public stills (Accessibility Barriers + Roboflow Sidewalk Barriers), not cane-camera / Pi Camera Module V2.
- Desktop GPU timings must not be treated as Raspberry Pi 5 performance.

## v0.2-exp2b

| Field | Value |
| --- | --- |
| Model version | `li-stick-navigation-v0.2-exp2b` |
| Checkpoint | `models/li_stick_navigation_v0.2_exp2b.pt` |
| Sidecar | `models/li_stick_navigation_v0.2_exp2b.class_map.yaml` |
| Dataset | `data/training/li_stick_v02_exp2b/` (reassigned splits; canonical provenance unchanged) |
| Training date | 2026-08-18 |
| Device | CUDA:0 RTX 4060 Laptop GPU (8 GB) |
| Image size / batch / epochs | 640 / 16 / 100 requested, 98 completed, early stop yes, best epoch 78 |
| Active training time | 4391.4 s (~73.2 min) |
| Held-out **test** | P=0.743 R=0.394 mAP50=0.356 mAP50-95=0.222 |
| pole (test n=43) | P=1.000 R=0.047 mAP50=0.045 mAP50-95=0.027 |
| bollard (test n=105) | P=0.589 R=0.314 mAP50=0.238 mAP50-95=0.106 |
| stairs (test n=158) | P=0.902 R=0.696 mAP50=0.677 mAP50-95=0.455 |
| curb (test n=98) | P=0.500 R=0.296 mAP50=0.214 mAP50-95=0.127 |
| barrier (test n=95) | P=0.829 R=0.716 mAP50=0.692 mAP50-95=0.507 |
| tree_trunk (test n=54) | P=0.640 R=0.296 mAP50=0.267 mAP50-95=0.110 |

Exp2b is **not** apples-to-apples with Exp2: the held-out split was rebuilt. Use Exp2b for trustworthy six-class evaluation.

### v0.2-exp2b known limitations

- Still a six-class **custom** detector. Does **not** emit COCO person/car/bicycle/etc.
- Pole recall is genuinely weak (0.047 on 43 test instances).
- Curb recall remains weak (~0.30).
- 1,418 pole rows remain PENDING and were not used.
- Images are public stills, not cane-camera / Pi Camera Module V2.


Production remains `python main.py` with `config/config.yaml` + `yolo26n.pt`.

Exp1 (stairs + curb):

```bash
python main.py --config config/config_v02_exp1.yaml --classes config/classes_v02_exp1.yaml
```

Exp2 (six custom hazards; no COCO classes):

```bash
python main.py --config config/config_v02_exp2.yaml --classes config/classes_v02_exp2.yaml
```

Exp2b (same six classes, group-aware stratified split):

```bash
python main.py --config config/config_v02_exp2b.yaml --classes config/classes_v02_exp2b.yaml
```

Webcam performance has not been measured in these experiments.
