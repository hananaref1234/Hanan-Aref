# Error analysis: li-stick-navigation-v0.2-exp1

Held-out **test** split. Ultralytics `val()` plus greedy IoU matching at conf=0.25, IoU=0.5.
Visual examples: `outputs/evaluation/v02_exp1/evaluation/{false_positive,false_negative}/{stairs,curb}/`.
This is not a safety review. Direction (up/down) was not evaluated.

## Counts (greedy match)

| Class | TP | FP | FN |
| --- | --- | --- | --- |
| stairs | 97 | 28 | 49 |
| curb | 32 | 30 | 71 |

## Confusion matrix (Ultralytics, test)

Rows = predicted, columns = true (`stairs`, `curb`, `background`):

| | stairs | curb | background |
| --- | --- | --- | --- |
| pred stairs | 99 | 1 | 26 |
| pred curb | 1 | 33 | 29 |
| pred background | 46 | 69 | 0 |

Normalized recall: stairs 0.68, curb 0.32. Stairs and curb are almost never swapped (1 box each way). The dominant error is **missed boxes** (background) and **extra boxes** on non-GT regions.

## Observed stairs false positives

Sampled `false_positive/stairs/` examples:

- Small **doorstep / 1–2 entrance steps** at storefronts (e.g. `test_000004738`, `test_000013373`).
- Extra or partial boxes on **real stairs** (duplicate localization / IoU miss against a different GT box), e.g. a top-landing crop (`test_000016988`) and a real outdoor stair (`test_000086405`).
- At least one box on **road / crosswalk pavement** rather than a stair (`test_000081482`).

Shelves, floor-tile patterns as stripes, and escalators were **not** observed in the sampled FP set.

## Observed stairs false negatives

- Stairs **far in the background** or behind signage (`test_000004912`).
- Very **small 1–2 step door entries** (`test_000006129`, `test_000013373`).

## Observed curb false positives

- Extra boxes on **real curbs** (duplicate or oversized relative to GT), including a walkway-corner curb (`test_000000780`) and a full sidewalk-to-road curb (`test_000005674`).
- A **curb cut / dropped curb** segment boxed separately from a large GT region (`test_000002086`).

Shadows, tile-only transitions, and road markings as the sole FP cause were **not** clearly isolated in the sampled set. One FP did include a white lane marking inside a curb-scale box (`test_000005674`), but the curb itself is present.

## Observed curb false negatives

- **Large ground-truth boxes** covering a whole paving change / dropped-curb zone (`test_000002086`) where the model fired a smaller box or missed the full extent.
- **Distant / small curb segments** at street corners with parked cars (`test_000002401`).

## Takeaway

The nano detector **does learn semantic stairs** on this public-photo set (test mAP50 0.635). **Curb is weak** (test mAP50 0.212, recall 0.29): many curbs are missed, and several FPs are extra boxes on real curbs rather than random objects. Domain is Accessibility Barriers stills, not a cane camera. No up/down. Not navigation-ready.
