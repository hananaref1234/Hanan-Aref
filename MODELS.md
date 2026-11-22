# Li-Stick AI — Model Registry

All trained checkpoints. Do not overwrite entries; append new rows.

---

## li-stick-navigation-v0.1

| Field | Value |
|---|---|
| File | `models/li_stick_navigation_v0.1.pt` |
| Experiment | Exp1 |
| Base model | YOLOv8n |
| Classes | 6 custom (pole, bollard, stairs, curb, barrier, tree_trunk) |
| Dataset | li_stick_v01 |
| mAP50 (val) | — |
| Notes | Initial prototype |

---

## li-stick-navigation-v0.2

| Field | Value |
|---|---|
| File | `models/li_stick_navigation_v0.2.pt` |
| Experiment | Exp2 |
| Base model | YOLOv8n |
| Classes | 6 custom |
| Dataset | li_stick_v02 |
| mAP50 (val) | — |
| Notes | Expanded dataset |

---

## li-stick-navigation-v0.2-exp2b

| Field | Value |
|---|---|
| File | `models/li_stick_navigation_v0.2_exp2b.pt` |
| Experiment | Exp2b |
| Base model | yolo26n.pt |
| Classes | 6 custom (pole, bollard, stairs, curb, barrier, tree_trunk) |
| Class map | `models/li_stick_navigation_v0.2_exp2b.class_map.yaml` |
| Dataset | li_stick_v02_exp2b |
| mAP50 (test) | see outputs/evaluation_exp2b |
| Notes | Canonical custom-only detector; frozen as Exp2b baseline |

---

## li-stick-navigation-v0.3-combined ← current

| Field | Value |
|---|---|
| File | `models/li_stick_navigation_v0.3_combined.pt` |
| Experiment | Exp3 (combined) |
| Base model | yolo26n.pt (transfer learning) |
| Classes | 13 — 6 custom hazards + 7 generic objects |
| Custom classes (IDs 0–5) | pole, bollard, stairs, curb, barrier, tree_trunk |
| Generic classes (IDs 6–12) | person, bicycle, car, motorcycle, bus, truck, chair |
| Class map | `models/li_stick_navigation_v0.3_combined.class_map.yaml` |
| Dataset | li_stick_v03_combined (9 780 train / 533 val / 618 test) |
| Training config | `config/training_v03_combined.yaml` |
| Webcam config | `config/config_v03_combined.yaml` |
| Epochs | 92 / 100 (early stop, patience 20) |
| Training time | ~20.9 h (RTX 4060 Laptop) |
| imgsz | 640 |
| batch | 16 |
| **mAP50 (test, all)** | **0.331** |
| mAP50-95 (test, all) | 0.215 |
| Notes | Single detector replacing separate custom+generic pipeline. Motorcycle, truck, chair recall weak — see eval report. |

### Per-class test metrics

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| pole | 98 | 0.417 | 0.153 | 0.098 | 0.047 |
| bollard | 142 | 0.585 | 0.268 | 0.214 | 0.099 |
| stairs | 50 | 0.818 | 0.540 | 0.523 | 0.332 |
| curb | 97 | 0.407 | 0.247 | 0.185 | 0.092 |
| barrier | 58 | 0.817 | 0.845 | 0.808 | 0.546 |
| tree_trunk | 192 | 0.491 | 0.141 | 0.094 | 0.038 |
| person | 665 | 0.876 | 0.466 | 0.458 | 0.262 |
| bicycle | 220 | 0.940 | 0.641 | 0.644 | 0.526 |
| car | 335 | 0.676 | 0.555 | 0.487 | 0.299 |
| motorcycle | 64 | 0.444 | 0.063 | 0.051 | 0.029 |
| bus | 41 | 1.000 | 0.463 | 0.465 | 0.376 |
| truck | 63 | 0.611 | 0.175 | 0.137 | 0.072 |
| chair | 86 | 0.552 | 0.186 | 0.143 | 0.074 |

Full report: `outputs/evaluation_v03_combined/`
