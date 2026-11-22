# Training view: li-stick-navigation-v0.3-combined

Derived Ultralytics dataset. Not the canonical taxonomy. Does not replace
`data/li_stick_navigation/`, Exp1, Exp2, or Exp2b.

## Training-specific class map

| Training ID | Name | Global Li-Stick ID |
| --- | --- | --- |
| 0 | pole | 0 |
| 1 | bollard | 1 |
| 2 | stairs | 2 |
| 3 | curb | 3 |
| 4 | barrier | 6 |
| 5 | tree_trunk | 11 |
| 6 | person | 12 |
| 7 | bicycle | 13 |
| 8 | car | 14 |
| 9 | motorcycle | 15 |
| 10 | bus | 16 |
| 11 | truck | 17 |
| 12 | chair | 18 |

## Counts at build time

| Split | Images | Annotations |
| --- | --- | --- |
| train | 9780 | 20391 |
| val | 533 | 1744 |
| test | 618 | 2111 |

See `pretrain_report.json` and `partial_label_analysis.json`.

## Partial-label strategy

Same-image merge of Roboflow generic labels onto approved custom Roboflow images; extra Roboflow images only if the source file has no pole/bollard/barricade/tree_trunk/stairs/curb labels; COCO chair images keep other selected COCO generics on the same image.

Recommendation: Partial labels are unavoidable for Accessibility Barriers. Do not drop those stairs/curb images. Prefer the same-image Roboflow merge (applied). Do not train a second detector. Monitor person/car false positives on indoor stairs photos. Future work: ignore-regions or person labels on Accessibility Barriers.
