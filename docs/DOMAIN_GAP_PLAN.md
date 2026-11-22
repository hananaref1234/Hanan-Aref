# Domain gap plan

Public datasets are useful for an **initial** Li-Stick detector. They are **not** a substitute for the cane camera.

Most candidate sources (SideGuide, Accessibility Barriers, DoorDetect, Cityscapes, typical pothole sets) were recorded with dashcams, phones, wheelchairs, or street-mapping rigs. None of them is the Raspberry Pi Camera Module V2 NoIR mounted on a cane, looking roughly chest-to-ground while walking.

That gap shows up as:

- different height and pitch
- cane swing and motion blur
- NoIR / IR-illuminated night appearance
- indoor corridors vs driving-view streets
- class definitions that do not match Li-Stick (`open` door vs traversable `open_doorway`; public `stair` is semantic `stairs`, not cane-camera geometry)

Public data alone is therefore **not** considered sufficient validation.

## Intended loop

```text
PUBLIC DATA  (licensed, mapped, reviewed)
    ↓
initial custom model  (later; not this step)
    ↓
Camera V2 NoIR testing on the real cane
    ↓
log failures (missed drop-offs, reversed stairs, overhead clutter, …)
    ↓
targeted Li-Stick capture of those failures
    ↓
fine-tune
    ↓
repeat
```

Do **not** try to recreate the entire public set with the cane. Collect where the public-trained model fails, plus a modest domain-calibration set (indoor/outdoor, day/night, stairs, curbs, doors, poles, trunks, walls).

First-party clips belong in `data/raw/listick_camera/` with session / video / location IDs so splits never leak.

## What this phase does not do

- Train a model
- Claim that imported public boxes will work on the cane
- Invent depth from bounding-box size
- Treat Cityscapes driving views as cane-height truth
