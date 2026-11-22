# Generic + custom class strategy

## Recommendation

**One lightweight model on Raspberry Pi 5**, trained with transfer learning from `yolo26n.pt`.

Do **not** run the pretrained COCO detector and a custom detector at the same time in production. That roughly doubles inference cost.

Fine-tuning **only** the custom classes replaces the detection head. Pretrained COCO outputs (person, car, chair, …) are **not** preserved by custom-only training. If those generics are required in the same Pi model, train a **combined** head.

## Taxonomy v0.2 detector IDs (active)

The detector classifies **semantic** hazards. It does not classify up vs down.

```text
0  pole
1  bollard
2  stairs
3  curb
4  open_doorway
5  closed_door
6  barrier
7  drop_off
8  pothole
9  low_overhead_obstacle
10 wall
11 tree_trunk
```

Taxonomy **v0.1** used `stairs_up` / `stairs_down` / `curb_up` / `curb_down` as detector classes. That version is preserved as documentation and as staging direction metadata. See `docs/TAXONOMY.md`.

## Combined IDs

Generics begin at **12**. Names match COCO (`chair`, not `custom_person`). v0.3-combined uses:

```text
12 person
13 bicycle
14 car
15 motorcycle
16 bus
17 truck
18 chair
```

See `config/classes_combined.yaml` and `COMBINED_CLASSES` in `src/dataset/taxonomy.py`.

The v0.3-combined **training** head is a contiguous 13-class subset (six custom hazards + these seven generics). Unused custom IDs 4, 5, 7–10 stay reserved and are not trained.

SideGuide person / bicycle / motorcycle / car / chair remain combined-taxonomy **candidates** only. SideGuide is not approved and must not be imported.

## Semantic vs geometric

```text
Camera
  → Object detector          (stairs / curb / pole / …)
  → Depth + geometry + time  (up / down, later)
  → Risk engine              (later)
```

Do not implement depth or geometry in this phase. The optional staging reviewer can still record `STAIRS_UP` etc. as research metadata. That review is **not** required for v0.2 detector training.

## Now

- Production `main.py` keeps `detector.model_path: yolo26n.pt` and `config/classes.yaml`.
- Public sources are listed in `data/sources/datasets.yaml`.
- `yolo26n.pt` has **no COCO `tree` class**. `tree_trunk` is custom ID 11.

## Rejected for production

- Two simultaneous YOLO models
- Retraining a huge detector for desktop mAP
- Renaming custom classes to match COCO
- Using COCO `potted plant` as a stand-in for `tree_trunk`
- Asking YOLO to learn `stairs_up` vs `stairs_down` from one RGB box
- Treating an “open door” folder as `open_doorway` without review
