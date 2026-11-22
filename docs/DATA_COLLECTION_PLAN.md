# Data collection plan

Goal: collect Li-Stick-specific footage with approximately the same camera geometry as the cane (Raspberry Pi Camera Module V2 NoIR, roughly chest-to-ground forward view, walking motion).

Public datasets are imported through `docs/DATASET_ACQUISITION.md` into `data/raw/` then `data/staging/`. They do not replace cane-camera data; see `docs/DOMAIN_GAP_PLAN.md`.

Do **not** collect dangerous data. Nobody should walk toward an unprotected drop, into traffic, or down unknown stairs to create training samples. Use public paths, indoor spaces you are allowed to film, and staged safe props.

## Quantity

There is no magic image count. For first experiments, aim for about **500–1000+ labeled instances per important custom class** where practical. Safety-critical classes (`drop_off`, `low_overhead_obstacle`) need more visual variety, not just more near-duplicate frames. `stairs` and `curb` are semantic detector classes; up/down is later geometry.

**Data diversity > raw image count.** Ten thousand frames from one corridor are not ten thousand examples.

## What to vary

- Indoor and outdoor
- Sidewalks, corridors, entrances, staircases
- Stair materials (concrete, wood, metal, tiled)
- Curb types (stone, concrete, painted, rounded)
- Daytime, evening, artificial lighting
- Shadows and backlighting
- Wet ground
- Crowded and empty scenes
- Narrow and wide passages
- Camera motion and cane swing
- Different approach angles
- Partial occlusion

## Video frames

Walking video will produce nearly identical consecutive frames. When extracting frames:

- Sample sparsely (for example 1–2 fps, or when the view actually changes)
- Keep `video_id`, `session_id`, `location_id`, `sequence_id`, `frame_index` in `metadata.csv`
- Put an entire video or sequence into **one** split (train or val or test)

`scripts/validate_dataset.py` flags leakage when metadata is present.

## Hard negatives

Record scenes that look like hazards but are not: striped floors, shelves, shadows, mirrors, windows. Save empty label files for those images. Label actual tree trunks as `tree_trunk`; do not leave them unlabeled as hard negatives.

## Privacy and rights

Avoid identifiable faces when possible. Do not scrape the web. Record license and permission in `DATASET_SOURCES.md` for any third-party clip.

## After capture

1. Extract frames into `images/train` (temporary holding is fine).
2. Annotate with the frozen class names in `ANNOTATION_GUIDE.md`.
3. Split by sequence, not by random frames.
4. Run `python scripts/validate_dataset.py` and `python scripts/dataset_stats.py`.
5. Only then consider `python scripts/train_custom_detector.py`.
