# Training view: li-stick-navigation-v0.2-exp2b

This directory is a **derived** Ultralytics dataset. It is not the canonical
Li-Stick taxonomy and must not replace `data/li_stick_navigation/`.

## Provenance

- Canonical dataset: `C:/Users/cezar/Desktop/li_stick_ai/data/li_stick_navigation`
- Dataset version: `li-stick-navigation-v0.2`
- Taxonomy version: `v0.2` (unchanged)
- Images are hardlinked or copied; labels are rewritten with training IDs.
- `metadata.csv` and `provenance.csv` are copies of the canonical sidecars.

## Why remap?

Ultralytics trains a head with contiguous class IDs starting at 0. The
project-wide taxonomy still uses the frozen v0.2 global IDs (0–11). Training
this experiment on the sparse 12-class YAML would create empty classes with
no labels. Empty classes are not invented here.

## Training-specific class map

| Training ID | Name | Global Li-Stick ID |
| --- | --- | --- |
| 0 | pole | 0 |
| 1 | bollard | 1 |
| 2 | stairs | 2 |
| 3 | curb | 3 |
| 4 | barrier | 6 |
| 5 | tree_trunk | 11 |

See `class_map.yaml`. The registered checkpoint sidecar
`models/<checkpoint>.class_map.yaml` uses the same mapping so
`ObjectDetector` emits global IDs and Li-Stick names.

## Classes not in this experiment

open_doorway, closed_door, drop_off, pothole, low_overhead_obstacle, wall

## Counts at build time

| Split | Images | Annotations |
| --- | --- | --- |
| train | 2605 | 4962 |
| val | 243 | 564 |
| test | 238 | 553 |

## Limitations

- Custom hazard classes only. No COCO/generic classes in this checkpoint.
- This experiment does not learn up/down. Direction is not a detector label.
- Not a safety-certified model.
