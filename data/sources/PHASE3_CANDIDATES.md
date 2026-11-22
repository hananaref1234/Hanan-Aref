# Phase 3 alternative sources (2026-08-18)

SideGuide is **not approved** and must not be used. This document records research for Exp2 data acquisition **without training**.

## Priority order researched

| Priority | Source | Exp2 utility | License status | Import status |
| --- | --- | --- | --- | --- |
| A | Open Images V7 | **None** — no pole/bollard/barrier/tree_trunk | Annotations CC BY 4.0; images CC BY 2.0 | **REJECTED** |
| B | TimberVision | Weak — forestry trunks, not sidewalk | CC BY-NC-SA 4.0 | **LICENSE_REVIEW_REQUIRED** |
| C | Roboflow Universe (vetted) | **Primary path** | Per-dataset | See below |
| — | SideGuide / Roboflow `scottsdale/sideguide` | Blocked | SideGuide terms | **REJECTED** |

## Open Images V7 findings

- **601** Ultralytics bbox classes; **600** in Google bbox taxonomy.
- **No** classes: pole, bollard, barrier, barricade, tree, tree_trunk.
- Near-miss labels (all **rejected** for auto-mapping):
  - `Tree` → not `tree_trunk`
  - `Street light` → not `pole`
  - `Traffic sign`, `Fire hydrant`, `Stop sign` → unrelated
- **Format:** CSV bbox + image IDs; FiftyOne/Ultralytics export to YOLO.
- **Attribution:** CC BY 4.0 annotations; CC BY 2.0 images with per-image verification caveat.
- **Action:** No download. Importer refuses conversion.

## TimberVision findings

- **~2k** RGB images, **~51k** trunk component masks (YOLOv8 seg + OBB).
- **License:** CC BY-NC-SA 4.0 — blocks commercial product use without legal review.
- **Semantics:** live/cut forestry logs — not pedestrian sidewalk `tree_trunk`.
- **Action:** Do not download until LICENSE_REVIEW_REQUIRED cleared.

## Roboflow Universe candidates

### Ready for manual download (CC BY 4.0 documented)

| Manifest name | URL | Images | Target classes | Mapping notes |
| --- | --- | --- | --- | --- |
| `roboflow_sidewalk_barriers` | [sidewalk-barriers](https://universe.roboflow.com/sidewalk-barriers-datasets/sidewalk-barriers) | ~24k | bollard, tree_trunk, barricade→barrier | Best multi-class sidewalk candidate |
| `roboflow_bollard_v2gn5` | [bollard-v2gn5](https://universe.roboflow.com/project-60htx/bollard-v2gn5) | ~1153 | bollard | Single-class supplement |
| `roboflow_power_pole` | [power-pole-msglw](https://universe.roboflow.com/power-pole-9gjpa/power-pole-msglw) | ~1.3k | Power-Pole → pole **review** | Utility poles — not auto-approved |

### Blocked pending license

| Manifest name | URL | Images | Issue |
| --- | --- | --- | --- |
| `roboflow_bollard_nkhwt` | [bollard-nkhwt](https://universe.roboflow.com/safe-walk-bd/bollard-nkhwt) | ~100 | License badge not shown |
| `roboflow_asset_model` | [asset_model-cvsgo](https://universe.roboflow.com/anpr-f7lrb/asset_model-cvsgo) | ~900 | License not shown |
| `roboflow_bollards_crosswalk_stairs` | [bollards-crosswalk-stairs](https://universe.roboflow.com/esera/bollards-crosswalk-stairs) | ~839 | License not shown |

### Rejected

| Manifest name | Reason |
| --- | --- |
| `roboflow_sideguide_scottsdale` | SideGuide-derived; SideGuide not approved |
| `sideguide` | Not approved for Li-Stick |

## After download workflow

```text
data/raw/<source>/     ← YOLO export untouched
scripts/import_source.py --source <name> --replace-source
data/staging/          ← PENDING review rows
scripts/review_staging.py
scripts/report_class_coverage.py --write data/sources/CLASS_COVERAGE.md
```

**Do not train Exp2** until class coverage is reviewed.

## Subsampling

Large Roboflow sources use deterministic whole-image subsampling (`max_instances_per_class: 2000`, `sample_seed: 42`).
