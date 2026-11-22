# Li-Stick custom class coverage (taxonomy v0.2)

Generated from staging provenance. Instance counts are semantic detector labels (`stairs`, `curb`, …). Direction is not a detector class in v0.2. Empty classes are reported as zero; they are not fabricated.

**Do not train until this snapshot is reviewed.**

## Coverage table

```text
CLASS                    IMAGES       INSTANCES    SOURCES                  STATUS

pole                     0            0            roboflow_sidewalk_barriers NO_DATA
bollard                  0            0            roboflow_sidewalk_barriers NO_DATA
stairs                   1217         1400         accessibility_barriers   GOOD
curb                     757          1123         accessibility_barriers   GOOD
open_doorway             0            0            -                        NO_DATA
closed_door              0            0            -                        NO_DATA
barrier                  0            0            roboflow_sidewalk_barriers NO_DATA
drop_off                 0            0            -                        NO_DATA
pothole                  0            0            -                        NO_DATA
low_overhead_obstacle    0            0            -                        NO_DATA
wall                     0            0            -                        NO_DATA
tree_trunk               0            0            roboflow_sidewalk_barriers NO_DATA
```

**Populated classes:** `stairs`, `curb`

**NO_DATA classes:** `pole`, `bollard`, `open_doorway`, `closed_door`, `barrier`, `drop_off`, `pothole`, `low_overhead_obstacle`, `wall`, `tree_trunk`

**Safety-critical classes still empty:** `drop_off`, `low_overhead_obstacle`

Direction review is not required for v0.2 object-detector training. Preserved v0.1 `STAIRS_*` / `CURB_*` values remain in staging `direction_status` / `direction_class` for later geometry research.
