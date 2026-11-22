# Dataset sources

Dataset version: **li-stick-navigation-v0.2** (`taxonomy_version: v0.2`)

Accessibility Barriers `stair` and curb-type `step` annotations are promoted as semantic `stairs` / `curb`. Direction labels from taxonomy v0.1 remain in staging/provenance metadata only.

The machine-readable manifest is [`data/sources/datasets.yaml`](../sources/datasets.yaml). Acquisition steps: [`docs/DATASET_ACQUISITION.md`](../../docs/DATASET_ACQUISITION.md).

Do not add a dataset to the training folders unless license, mapping, and review status are recorded.

## Recorded sources

| Dataset name | Source | License | Status | Mapping | Date accessed |
| --- | --- | --- | --- | --- | --- |
| SideGuide | https://ytaek-oh.github.io/sideguide | SideGuide terms (non-commercial, no redistribution). **NOT approved for Li-Stick.** | MANUAL_DOWNLOAD_REQUIRED / blocked | Do not import | 2026-08-18 |
| Open Images V7 | https://storage.googleapis.com/openimages/web/index.html | CC BY 4.0 annotations; CC BY 2.0 images | REJECTED for Exp2 targets | No pole/bollard/barrier/tree_trunk classes | 2026-08-18 |
| TimberVision | https://zenodo.org/records/14825846 | CC BY-NC-SA 4.0 | LICENSE_REVIEW_REQUIRED | Forestry domain; not sidewalk tree_trunk | 2026-08-18 |
| Roboflow Sidewalk Barriers | https://universe.roboflow.com/sidewalk-barriers-datasets/sidewalk-barriers | CC BY 4.0 | MANUAL_DOWNLOAD_REQUIRED | custom: bollard, tree_trunk, barricade→barrier; v0.3 generics: person (incl. pedestrian), bicycle, car, motorcycle, bus, truck. No chair. jeep/van/cone not mapped. | 2026-08-18 |
| Roboflow bollard v2gn5 | https://universe.roboflow.com/project-60htx/bollard-v2gn5 | CC BY 4.0 | MANUAL_DOWNLOAD_REQUIRED | bollard | 2026-08-18 |
| Roboflow Power Pole | https://universe.roboflow.com/power-pole-9gjpa/power-pole-msglw | CC BY 4.0 | MANUAL_DOWNLOAD_REQUIRED | utility pole → semantic review | 2026-08-18 |
| Accessibility Barriers | https://zenodo.org/records/6382090 | CC-BY-4.0 | MANUAL_REVIEW_REQUIRED | Taxonomy v0.2: `stair` → `stairs`, curb-type `step` → `curb`. doorstep/other/ramp/grab_bar unmapped. v0.1 direction decisions preserved as metadata. | 2026-08-17 |
| DoorDetect | https://github.com/gasparramoa/DoorDetect-Class-Dataset | unspecified on GitHub | LICENSE_REVIEW_REQUIRED | open/closed → semantic review; semi-open unmapped | 2026-08-17 |
| Cityscapes | https://www.cityscapes-dataset.com/ | non-commercial, no redistribution | LICENSE_REVIEW_REQUIRED | wall/pole mask→box; never vegetation→tree_trunk | 2026-08-17 |
| Pothole (unselected) | — | — | MANUAL_REVIEW_REQUIRED | pothole → pothole (10) after a licensed set is chosen | 2026-08-17 |
| Mapillary Vistas | https://www.mapillary.com/dataset/vistas | CC-BY-NC-SA + research-use terms | LICENSE_REVIEW_REQUIRED | not integrated | 2026-08-17 |
| COCO 2017 (filtered subset) | http://images.cocodataset.org/ | annotations CC BY 4.0; images Flickr (per-image) | MANUAL_DOWNLOAD_REQUIRED | v0.3-combined chair source; other selected generics on the same images kept | 2026-08-18 |

## Notes

- Inspected `yolo26n.pt` class names (COCO 80): there is **no `tree` class**. `tree_trunk` is custom ID 11. Do not import COCO `potted plant` as a tree mapping.
- SideGuide is **not approved** for Li-Stick. Do not import unless licensing is later approved.
- Phase 3 Roboflow CC BY 4.0 candidates require manual Universe download — see `data/sources/PHASE3_CANDIDATES.md`.
- `drop_off` and `low_overhead_obstacle` have no adequate public source in this list.
- Detector classes are semantic (`stairs`, `curb`). Direction is later geometry, not a YOLO label. Traversable doorways still require human review.

## v0.3-combined generic sources (2026-08-18)

### Roboflow Sidewalk Barriers (CC BY 4.0)

- Official Universe page; already on disk from Exp2 custom import.
- Selected generic classes: person (including source `pedestrian` → person), bicycle, car, motorcycle, bus, truck.
- Not used: jeep, van, cone, chair (absent), and other unrelated classes.
- Filtering: same-image merge onto approved custom Roboflow images first; additional generic-only images only if the source file has no pole/bollard/barricade/tree_trunk/stairs/curb labels. Whole-image instance caps (seed 42).
- Combined-view contribution: 17,473 generic+custom boxes from this source (see `data/training/li_stick_v03_combined/pretrain_report.json`).

### COCO 2017 filtered subset (official images.cocodataset.org)

- Date accessed: 2026-08-18
- License: annotations CC BY 4.0; images are Flickr photos with per-image licenses
- Selected classes: person, bicycle, car, motorcycle, bus, truck, chair
- Filtering method: official `instances_train2017.json`; skip crowd boxes; whole-image sample of chair images until 2000 chair instances (seed 42). Other selected generic boxes on those images are kept.
- Image count: 830
- Annotation count: 4250 (chair 2000, person 1692, car 214, bicycle 130, motorcycle 115, truck 64, bus 35)
- Stored at `data/raw/coco_2017_generic_subset/`
