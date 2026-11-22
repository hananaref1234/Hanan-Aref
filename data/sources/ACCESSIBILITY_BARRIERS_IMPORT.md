# Accessibility Barriers import (Zenodo 6382090)

Downloaded **2026-08-17** from the official record only:
https://zenodo.org/records/6382090

License: **CC BY 4.0** (Zenodo `license.id = cc-by-4.0`), re-verified before download.

## Files

| Item | Value |
| --- | --- |
| Archive | `data/raw/accessibility_barriers/wm_barriers_data.zip` |
| Download size | 7,972,407,510 bytes |
| MD5 | `d96051ca1339d1c1afa9ac736eabd058` (verified) |
| Extracted | `images/` (5,538 files) + `wm_annotations.xml` |
| Extracted images+xml | ~8.00 GB |
| Annotation format | CVAT XML, bounding boxes + attributes |

## Observed source taxonomy (from XML, not estimates)

| Label | Count | Li-Stick handling (taxonomy v0.2) |
| --- | --- | --- |
| `stair` | 1,492 | detector class `stairs` (DIRECT) |
| `step` type=`curb` | 1,185 | detector class `curb` (DIRECT) |
| `step` type=`doorstep` | 2,128 | `UNMAPPED_SOURCE_CLASS` |
| `step` type=`other` | 251 | `UNMAPPED_SOURCE_CLASS` |
| `ramp` | 143 | `UNMAPPED_SOURCE_CLASS` |
| `grab_bar` | 922 | `UNMAPPED_SOURCE_CLASS` |
| **Total annotations** | **6,121** | |
| **Total images** | **5,538** | 0 missing |

Direction is **not** a detector label. v0.1 `STAIRS_*` / `CURB_*` decisions are preserved as `direction_status` / `direction_class`. Only `step` with `type=curb` maps to `curb`. doorstep/other/ramp/grab_bar stay unmapped.

## Staging (v0.2)

Rebuild with:

```text
python scripts/migrate_taxonomy_v02.py
```

v0.1 metadata is copied to `data/staging/staging_metadata_v0.1_directional.csv` before replacement.

## Split strategy

The source has **no original train/val/test split**. Images are independent public-space photos (Wheelmap), not a walking video.

Promotion assigns a seeded 80/10/10 split by unique `source_image` (seed 42):

- every box from the same `source_image` stays in one split
- no image appears in more than one split

## Direction review (optional)

Direction review is **not** required for v0.2 object-detector training.

The reviewer remains for later geometry research:

```text
python scripts/review_staging.py --kind stairs
python scripts/review_staging.py --kind curb
```

Keys: `1` stairs_up, `2` stairs_down, `3` curb_up, `4` curb_down, `r`/`x` reject, `s` skip, `q` quit. Detector class stays `stairs` or `curb`.
