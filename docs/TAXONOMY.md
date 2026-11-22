# Taxonomy versions

Li-Stick does **not** silently rewrite a frozen taxonomy. Detector class IDs are versioned.

## v0.1 (historical)

The first custom detector taxonomy treated level-change **direction** as a class:

```text
0  pole
1  bollard
2  stairs_up
3  stairs_down
4  curb_up
5  curb_down
6  open_doorway
7  closed_door
8  barrier
9  drop_off
10 pothole
11 low_overhead_obstacle
12 wall
13 tree_trunk
```

Human review wrote `STAIRS_UP` / `STAIRS_DOWN` / `CURB_UP` / `CURB_DOWN` into staging. Those decisions are archived in `data/staging/staging_metadata_v0.1_directional.csv` and copied onto v0.2 rows as `direction_status` / `direction_class`. They are **not** detector labels anymore.

## v0.2 (active)

The object detector answers “what is present?”, not “does it go up or down?”.

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

Generics in the combined head start at **12** (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `chair`).

| v0.1 detector class | v0.2 detector class |
| --- | --- |
| stairs_up | stairs |
| stairs_down | stairs |
| curb_up | curb |
| curb_down | curb |

Direction (`stairs_up` vs `stairs_down`, `curb_up` vs `curb_down`) is reserved for a later depth / geometry / temporal module. It is not trained into YOLO.

See `src/dataset/taxonomy.py`, `docs/CLASS_STRATEGY.md`, and `docs/PERCEPTION_FEASIBILITY.md`.
