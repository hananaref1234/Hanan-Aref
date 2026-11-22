# Dataset acquisition

This is the human-facing procedure for building Li-Stick navigation data from public sources. It does **not** train a model.

Pipeline:

```text
request / download (legitimate only)
    ↓
data/raw/<source>/          original files, untouched
    ↓
scripts/import_source.py    safe class mapping → staging
    ↓
data/staging/               converted samples + staging_metadata.csv
    ↓
scripts/review_staging.py   human approve / reject / direction
    ↓
scripts/promote_staging.py  approved samples only
    ↓
data/li_stick_navigation/   frozen YOLO training set
```

Do not copy third-party archives straight into `images/train`.

## Manifest

[`data/sources/datasets.yaml`](../data/sources/datasets.yaml) is the source of truth. No source is `APPROVED` until both license and mapping are verified.

Statuses: `APPROVED`, `MANUAL_DOWNLOAD_REQUIRED`, `MANUAL_REVIEW_REQUIRED`, `LICENSE_REVIEW_REQUIRED`, `REJECTED`.

## Commands

```text
python scripts/inspect_sources.py
python scripts/inspect_sources.py --source sideguide --instructions
python scripts/prepare_source_download.py --source accessibility_barriers
python scripts/import_source.py --source sideguide
python scripts/report_accessibility_barriers.py
python scripts/validate_staging.py
python scripts/migrate_taxonomy_v02.py
python scripts/report_class_coverage.py --write data/sources/CLASS_COVERAGE.md
```

`prepare_source_download.py --download` is opt-in and blocked for SideGuide, Cityscapes, Mapillary Vistas, DoorDetect, and pothole. It will not bypass access forms.

SideGuide re-inspected **2026-08-18**: `data/raw/sideguide/` is README-only. Status remains `MANUAL_DOWNLOAD_REQUIRED`. **SideGuide is NOT approved for Li-Stick** — do not import.

Phase 3 alternatives documented in [`data/sources/PHASE3_CANDIDATES.md`](../data/sources/PHASE3_CANDIDATES.md). CC BY 4.0 Roboflow sources require manual Universe download before import:

```text
python scripts/inspect_sources.py --source roboflow_sidewalk_barriers --instructions
python scripts/import_source.py --source roboflow_sidewalk_barriers --replace-source
```

Do not start Exp2 training until staging is reviewed and class coverage is acceptable.

## Where to put downloads

| Source | Place files here |
| --- | --- |
| SideGuide | `data/raw/sideguide/` |
| Accessibility Barriers | `data/raw/accessibility_barriers/` |
| DoorDetect | `data/raw/doordetect/` |
| Cityscapes | `data/raw/cityscapes/leftImg8bit` and `gtFine` |
| Pothole (after selecting one licensed set) | `data/raw/pothole/` |
| First-party cane camera | `data/raw/listick_camera/` |
| Mapillary Vistas | do not download |

## Mapping rules that code will not violate

- `stairs` / `step` / `curb` never become `stairs_up`, `stairs_down`, `curb_up`, or `curb_down` automatically. Taxonomy v0.2 maps `stair` → `stairs` and curb-type `step` → `curb` for the **detector** only.
- `open` door never becomes `open_doorway` automatically.
- `vegetation` never becomes `tree_trunk`.
- SideGuide `barricade` is only a candidate for `barrier` after semantic review.
- Cityscapes boxes from masks are `MASK_TO_BOX` and stay in review because the conversion is lossy.

## Combined taxonomy

Custom IDs 0–11 stay frozen (taxonomy v0.2). Generics start at 12 (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `chair`). Importing generics is opt-in (`--include-generic`). Fine-tuning only the custom classes would replace the pretrained COCO head; v0.3-combined trains one 13-class detector on labeled custom + selected generic annotations.

See `docs/TAXONOMY.md`, `docs/CLASS_STRATEGY.md` and `docs/DOMAIN_GAP_PLAN.md`.
