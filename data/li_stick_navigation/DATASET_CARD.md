# Dataset card: li-stick-navigation-v0.2

| Field | Value |
| --- | --- |
| Dataset version | li-stick-navigation-v0.2 |
| Taxonomy version | v0.2 |
| Task | Bounding-box object detection (YOLO) |
| Classes | 12 frozen custom navigation classes (IDs 0–11; `wall` = 10, `tree_trunk` = 11) |
| Current labeled images | 1759 (Accessibility Barriers stills) |
| Current labeled instances | 2523 (`stairs` 1400, `curb` 1123) |
| Splits | train 1407 images / 2036 boxes; val 176 / 238; test 176 / 249 (seed 42, 80/10/10 by source image) |
| Collection geometry | Raspberry Pi Camera Module V2 NoIR on a cane (target). Current images are public Accessibility Barriers photos, not cane-camera. |
| Status | Semantic `stairs` / `curb` promoted from Accessibility Barriers. **Do not train until this snapshot is reviewed.** |

A trained model must record this version string in `models/MODELS.md`.

## Changelog from v0.1

- Detector classes `stairs_up` / `stairs_down` / `curb_up` / `curb_down` were replaced by `stairs` and `curb`.
- v0.1 human direction decisions are preserved as `direction_status` / `direction_class`, not as YOLO labels.
- Direction review is not required for detector training.

## Intended use

Training and evaluating Li-Stick **semantic** hazard perception. Not a certified safety dataset. Up/down is out of scope for this detector version.

## Out of scope

- Generic COCO classes (person, car, chair, …) live in the pretrained Phase 1 model until they are labeled here as IDs 12+
- `tree_trunk` is a custom class (ID 11). COCO/`yolo26n.pt` has no `tree` class; do not map `potted plant` onto it
- Depth, segmentation, and up/down state are not part of v0.2 detector labels

## Split policy

Do not randomly split consecutive video frames. Group by `video_id` / `session_id` / `sequence_id` using `metadata.csv`. Accessibility Barriers still photos have no video id; all boxes from the same `source_image` stay in one split.
