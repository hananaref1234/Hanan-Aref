# Perception feasibility (bbox detection vs later sensors)

This is an engineering recommendation. Nothing here is implemented except bounding-box detection.

Class names in taxonomy **v0.2** are semantic detector labels. Up vs down is **not** a YOLO class.

| Class | Initial recommendation | Why |
| --- | --- | --- |
| pole | object detection | Distinct vertical **man-made** instance; bbox is reasonable. Do not use for tree trunks. |
| bollard | object detection | Short posts are box-friendly. Keep separate from pole. |
| stairs | object detection | A box can find a staircase region. **Up vs down is not decided from one RGB box.** |
| curb | object detection | Thin, view-dependent edge. Direction needs geometry/depth later. |
| open_doorway | hybrid: detect the opening (not the door leaf) | The traversable hole is not a typical “object.” Mirrors/windows confuse RGB. |
| closed_door | object detection | A closed architectural door is a reasonably boxable object. |
| barrier | object detection | Construction barriers and barricades are box-friendly. Keep separate from `wall`. |
| drop_off | hybrid: detection is insufficient; need depth / geometry / sensors | Monocular RGB easily confuses shadows, mats, and texture with a void. **Do not trust bbox-only drop_off.** |
| pothole | object detection, later depth | Larger holes can be boxed; small ones need more surface geometry. |
| low_overhead_obstacle | object detection + later depth | Cane-missable; bbox can locate the object, not the clearance. Branches/signs/beams, not whole trees or walls. |
| wall | object detection or later segmentation | Architectural surface that bounds a path. Large and often truncated; a box is a coarse start. |
| tree_trunk | object detection | Vertical woody stem that can obstruct walking. Box the trunk, not the canopy. **Not in yolo26n.pt / COCO.** Do not use `potted plant` or `pole`. |

## Semantic perception vs geometric reasoning

```text
Camera
  ↓
Object detector (YOLO)
  ↓
stairs / curb / other custom classes
  ↓
Depth + geometry + temporal information   (not implemented)
  ↓
up / down state
  ↓
risk engine                               (not implemented)
```

YOLO/object detector:

- detects `stairs`
- detects `curb`

Future geometry / depth / temporal module:

- determines `stairs_up` vs `stairs_down` where possible
- determines `curb_up` vs `curb_down` where possible

Risk engine will consume both later. Do **not** implement depth or geometry in this phase.

Preserved v0.1 human direction labels remain in staging metadata for that later work.

## stairs

Can visual bounding-box detection reliably distinguish direction? **No.** Direction depends on camera pitch, approach, and which landing is the walking surface. Use bbox to find *a staircase*. Annotators label `stairs`, not up/down.

## drop_off

Can monocular object detection reliably distinguish a physical drop from a shadow or texture change? **No, not reliably.** Treat this class as experimental. Sensor fusion (ultrasonic / depth) belongs in a later phase. Do not invent distance from box size.

## curb

Would segmentation or depth improve detection? **Yes.** Curbs are thin structures. Keep one detector class (`curb`). Up vs down is geometric, not a second YOLO class.

## open_doorway

Should the system detect the physical door, the opening, or both? **The usable opening** (`open_doorway`) and a blocking leaf (`closed_door`) are different navigation events. Do not box the whole wall. Later geometric analysis (vanishing lines, depth hole) may outperform a box around “empty space.”

## Depth field

`Detection.depth_m` is reserved as `None` until a real depth source exists. Bounding-box size is not distance.
