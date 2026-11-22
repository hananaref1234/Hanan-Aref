# Raw SideGuide

**Status (inspected 2026-08-18): NOT PRESENT.** This folder contains only this README. There are 0 images, 0 annotation files, and 0 archives.

Do not scrape or bypass the request form. Do not flatten files into `data/li_stick_navigation/`. Never modify official files after they arrive.

## Why Exp2 is blocked

`li-stick-navigation-v0.2-exp2` needs SideGuide samples for `pole`, `bollard`, `tree_trunk`, and candidate `barrier` (`barricade` after review). Those files are not on disk. The Accessibility Barriers stairs/curb data is already in the project and must not be used as a substitute for SideGuide classes.

## Official access (human action required)

1. Open the project page: https://ytaek-oh.github.io/sideguide
2. Complete the authors' access form (academic email required):
   https://docs.google.com/forms/d/e/1FAIpQLScBmoVoj0d-omBOVCHGjhRislXP0TYzRqaUJOmJcqN6ylQcxQ/viewform
3. Form fields: Name, academic Email, Affiliation, Supervisor's Name, Purpose (in detail), Homepage (optional), Agreement to terms.
4. Wait for approval. The authors email a download link. Questions: `datahub@aiworkx.ai` (also `datahub@testworks.co.kr`).
5. Extract the **official** archive into this directory (`data/raw/sideguide/`). Keep original folder names and annotations intact.
6. Then run:

```text
python scripts/inspect_sources.py --source sideguide
```

Do **not** download from unofficial mirrors. Do **not** scrape AI-Hub (`aihub.or.kr`) or bypass Korean-only access. Automatic download is disabled in this repo.

## License / terms (from the official form, 2026-08-18)

The form is titled "SideGuide Terms of Use" (Testworks, Inc.). Key points:

- Non-commercial research use only.
- Academic address required on the form.
- No copy, duplicate, reproduce, assign, distribute, publish, sell, sublicense, or trade of the dataset **or derived data**.
- Korean law; authors may terminate access.
- Cite Park, Oh, Ham, Joo, Kim, Kum, Kweon, IROS 2020.

If Li-Stick use is commercial, get a legal decision **before** requesting the data. This repo does not treat SideGuide as `APPROVED`.

## Expected official contents (published; archive names unknown until the download link arrives)

Prefer the **bounding-box set** for Exp2. Polygon masks and stereo disparity are not required for this experiment.

Published scale (paper / Testworks datasheet):

| Split | BBox images (published) | BBox instances (published) |
| --- | --- | --- |
| train | 249K | 2,379K |
| val | 37K | 351K |
| test | 67K | 652K |
| total | ~350K | ~3.38M |

Published CSV bbox fields: image id, name, width, height, label, occluded, xtl/ytl, xbr/ybr, z_order.

Published static classes that matter for Exp2: `pole`, `bollard`, `tree trunk`, `barricade`. Do not treat `potted plant` / vegetation as `tree_trunk`. Person, car, bicycle, and other generics must not enter Exp2.

The current importer (`src/dataset/importers/sideguide.py`) parses JSON. If the official release is CSV (as published), add CSV parsing after the files are on disk. Do not invent a fake JSON conversion beforehand.

## After a legitimate download

Leave originals here. Convert only through `scripts/import_source.py --source sideguide` into `data/staging/`. Then continue Exp2.
