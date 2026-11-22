# Staging review

Ambiguous public labels are not written into the training set until a person looks at them.

Queue file: `data/staging/staging_metadata.csv`

Tool:

```text
python scripts/review_staging.py --list
python scripts/review_staging.py --kind stairs
python scripts/review_staging.py --kind curb
python scripts/review_staging.py --kind pole
python scripts/review_staging.py --kind bollard
python scripts/review_staging.py --kind tree_trunk
python scripts/review_staging.py --kind barrier
python scripts/review_staging.py --limit 20
python scripts/review_staging.py --stats --kind pole
```

**Direction review is NOT required for taxonomy v0.2 object-detector training.**

`stair` → `stairs` and curb-type `step` → `curb` are semantic mappings. The reviewer remains as an optional research tool for later depth / geometry / temporal work. Keys `1`–`4` write `direction_status` / `direction_class` and keep the detector class as `stairs` or `curb`.

v0.1 human decisions are archived in `data/staging/staging_metadata_v0.1_directional.csv`.

## Keyboard

### Semantic review (`pole`, `bollard`, `tree_trunk`, `barrier`)

| Key | Action |
| --- | --- |
| `a` | `APPROVED` (uses `mapped_class` or `proposed_class`) |
| `r` | `REJECTED` |
| `n` / `s` | skip / next, no change |
| `p` | previous |
| `h` | print help |
| `q` | quit and save |

### Direction review (`stairs`, `curb`)

| Key | Action |
| --- | --- |
| `a` | `APPROVED` (uses `mapped_class` or `proposed_class`) |
| `r` | `REJECTED` |
| `1` | `STAIRS_UP` direction metadata; detector class `stairs` |
| `2` | `STAIRS_DOWN` direction metadata; detector class `stairs` |
| `3` | `CURB_UP` direction metadata; detector class `curb` |
| `4` | `CURB_DOWN` direction metadata; detector class `curb` |
| `x` | `AMBIGUOUS` (not promoted) |
| `n` / `s` | skip / next, no change |
| `p` | previous |
| `h` | print help |
| `q` | quit and save |

Do **not** guess stairs or curb direction if you are using this tool for research metadata. If you cannot see whether the user would go up or down, press `x`.

For semantic Phase 3 review:

- approve `pole` only for narrow man-made vertical poles/posts
- approve `bollard` only for true bollards
- approve `tree_trunk` only for the main trunk / vertical stem
- approve `barrier` only when source `barricade` really blocks pedestrian travel

## What gets promoted

`scripts/promote_staging.py` copies semantic detector classes:

- `APPROVED` with a v0.2 class (`stairs`, `curb`, `pole`, …)
- preserved `STAIRS_UP` / `STAIRS_DOWN` → detector `stairs`
- preserved `CURB_UP` / `CURB_DOWN` → detector `curb`

`PENDING`, `NEEDS_DIRECTION_REVIEW`, `REJECTED`, and `AMBIGUOUS` stay in staging.

Each `source_image` must remain in a single split (`train` or `val` or `test`). Original dataset splits are kept when present. Unsplit sources use a seeded 80/10/10 split by unique source image.

## Mapping types

| Type | Meaning |
| --- | --- |
| `DIRECT` | Safe name match. SideGuide `pole` still starts as `PENDING`. Accessibility Barriers `stair`/`curb` are semantic `APPROVED` in v0.2. |
| `MANUAL_DIRECTION` | Optional research direction labels; not required for detector training |
| `SEMANTIC_REVIEW` | Meaning differs (open door vs doorway, barricade vs barrier). Phase 3 `barrier` review uses this path. |
| `MASK_TO_BOX` | Semantic mask collapsed to a rectangle; information is lost |
| `CUSTOM` | Source-specific / unmapped |
