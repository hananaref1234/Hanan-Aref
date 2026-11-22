# Li-Stick annotation guide

Dataset version: **li-stick-navigation-v0.2** (`taxonomy_version: v0.2`)

Use these class names exactly. Do not rename them in labels, YAML, or tools.

Format: YOLO detection labels, one `.txt` per image, one line per box:

```text
class_id x_center y_center width height
```

All four geometry values are normalized to `[0, 1]`.

If an image contains no custom hazards (hard negative), keep an **empty** `.txt` file with the same stem as the image.

## Global rules

- Annotate navigation-relevant objects that a cane user could collide with, fall from, or fail to notice.
- The detector class for a staircase is `stairs`. The detector class for a curb is `curb`. Do **not** label `stairs_up` / `stairs_down` / `curb_up` / `curb_down` for YOLO. Direction is later geometry, not a bounding-box class.
- If you cannot name the semantic class confidently, do not guess. Add the image to a review list instead of inventing doorway state or `drop_off`.
- Multiple hazards in one image: draw every distinct instance.
- Partially occluded objects: annotate if the object is still identifiable and at least about 40% of the obstacle is visible.
- Truncated at the image border: annotate the visible portion. Do not invent the off-screen extent.
- Tiny or distant instances that would not affect walking in the next few meters: skip.
- Do not label reflections, posters, or screens of an object as the object.
- Indoor and outdoor scenes both count. Night / low-light images are valuable; only label what you can actually see.
- The cane camera will move. Slight motion blur is acceptable if the class is still clear.

## pole

Use these class names exactly. Do not rename them in labels, YAML, or tools.

Format: YOLO detection labels, one `.txt` per image, one line per box:

```text
class_id x_center y_center width height
```

All four geometry values are normalized to `[0, 1]`.

If an image contains no custom hazards (hard negative), keep an **empty** `.txt` file with the same stem as the image.

## Global rules

- Annotate navigation-relevant objects that a cane user could collide with, fall from, or fail to notice.
- If you cannot name the class confidently, do not guess. Add the image to a review list instead of inventing `stairs_up` vs `stairs_down`, `curb_up` vs `curb_down`, doorway state, or `drop_off`.
- Multiple hazards in one image: draw every distinct instance.
- Partially occluded objects: annotate if the object is still identifiable and at least about 40% of the obstacle is visible.
- Truncated at the image border: annotate the visible portion. Do not invent the off-screen extent.
- Tiny or distant instances that would not affect walking in the next few meters: skip.
- Do not label reflections, posters, or screens of an object as the object.
- Indoor and outdoor scenes both count. Night / low-light images are valuable; only label what you can actually see.
- The cane camera will move. Slight motion blur is acceptable if the class is still clear.

## pole

Counts: narrow vertical **man-made** posts a pedestrian could walk into — utility poles, lamp posts, sign posts, structural posts.

Does not count: tree trunks (`tree_trunk`), people, chair legs, door frames, architectural walls (`wall`). Keep separate from `bollard`.

## bollard

Counts: short pedestrian or vehicle-control posts — sidewalk bollards, parking bollards, protective posts.

Does not count: tall lamp or utility poles (`pole`), people.

## stairs

Counts: a staircase or meaningful sequence of steps. Draw one box around the visible staircase region, not each step.

Does not count: a single curb (`curb`), escalators, ladders, striped floors, shelves, shadows of steps, ramps without steps.

Do not encode up vs down in the detector label. Optional research metadata may still record direction separately.

## curb

Counts: a curb or single step-height level change at the walking surface.

Does not count: full staircases (`stairs`), painted lines, tiny pavement lips that are not a real step, large platform edges (`drop_off`).

Do not encode up vs down in the detector label. Optional research metadata may still record direction separately.

## open_doorway

Counts: a traversable opening — open door, doorway with the door fully open, entrance or passage opening. Annotate the usable opening, not the entire wall.

Does not count: windows, mirrors, dark rectangles on walls, closed doors (`closed_door`).

## closed_door

Counts: a navigation-relevant architectural door that blocks an opening.

Does not count: cabinet doors, appliance doors, furniture, decorative panels.

## barrier

Counts: physical barriers restricting a walking path — construction barriers, fence sections across a route, pedestrian barricades, roadwork barriers.

Does not count: distant perimeter fences that do not affect the current path.

## drop_off

Counts: a significant visible edge where continuing forward could mean a dangerous downward drop — platform edge, unprotected level change, edge of a raised walkway.

Does not count: normal shadows, black mats, wet patches, floor color changes, small curbs (`curb`).

This class is difficult. Do not claim it is reliable until evaluation says so. If unsure, skip and flag.

## pothole

Counts: a meaningful depression or damaged walking or road surface that could trip a pedestrian.

Does not count: hairline cracks, painted patches, flush manhole covers, tiny pits.

## low_overhead_obstacle

Counts: objects in the upper-body or head path that a ground cane may miss — low branches, projecting signs, low beams, hanging objects.

Does not count: high ceilings, objects the user would not walk under, a tree trunk in the walking path (`tree_trunk`), a full architectural wall (`wall`). Low branches and hanging vegetation in the head path stay in this class.

## wall

Counts: an architectural wall or wall-like barrier surface that is relevant to navigation — interior walls, exterior building walls, retaining walls that block or bound a walking path.

Does not count: narrow man-made posts (`pole`), short control posts (`bollard`), temporary construction barricades (`barrier`), closed doors (`closed_door`), tree trunks (`tree_trunk`). Do not box an entire room; label the wall surface that actually constrains the path.

## tree_trunk

Counts: the trunk or main vertical woody stem of a tree that can physically obstruct the walking path. Box the trunk, not the whole canopy.

Does not count:

- the entire tree canopy or foliage mass
- man-made posts (`pole`)
- short control posts (`bollard`)
- architectural walls (`wall`)
- branches or vegetation that only intrude into the head/upper-body path (`low_overhead_obstacle`)
- COCO `potted plant` (not a Li-Stick class; the pretrained model has no `tree` class)

A trunk that is clearly off the walking path may still be labeled if it is visible. Relevance and risk filtering are later phases, not part of annotation.

The pretrained `yolo26n.pt` checkpoint has no `tree` class. Do not map `potted plant` onto `tree_trunk`.

## Hard negatives

Include scenes that look like a class but are not. Those images may have empty labels, or labels only for other true classes.

- stairs: striped floors, shelves, horizontal lines, escalators, shadows, tiled floors
- drop_off: dark shadows, black mats, reflections, floor color changes
- pole: tree trunks (`tree_trunk` instead), people, thin furniture, door frames, walls
- tree_trunk: whole canopies, hedges, poles, potted plants
- wall: curtains, fences (`barrier`), hedges, distant building facades that do not affect the current path
- open_doorway: mirrors, dark walls, windows, painted rectangles

## Ambiguous samples

Keep a review list instead of forcing a class, especially for drop_off vs shadow and open vs closed doorway. Stairs/curb **direction** is not a detector annotation in v0.2.
