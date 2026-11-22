# Training view: li-stick-navigation-v0.2-exp1

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
project-wide taxonomy still uses:

- stairs = global ID 2
- curb = global ID 3

Training this experiment on the sparse 12-class YAML would create empty
classes (pole, drop_off, …) with no labels. Empty classes are not invented
here. Only stairs and curb are trained.

## Training-specific class map

| Training ID | Name | Global Li-Stick ID |
| --- | --- | --- |
| 0 | stairs | 2 |
| 1 | curb | 3 |

See `class_map.yaml`. The registered checkpoint sidecar
`models/<checkpoint>.class_map.yaml` uses the same mapping so
`ObjectDetector` emits global IDs and Li-Stick names.

## Counts at build time

| Split | Images | Annotations |
| --- | --- | --- |
| train | 1407 | 2036 |
| val | 176 | 238 |
| test | 176 | 249 |

## Limitations

- No training data for pole, bollard, doors, barrier, drop_off, pothole,
  low_overhead_obstacle, wall, or tree_trunk.
- This experiment does not learn up/down. Direction is not a detector label.
- Not a safety-certified model.
