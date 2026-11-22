# Dataset and hardware handoff — requested from Li-Stick team

**Missing**: actual annotated train/validation/test images, original v0.3 checkpoint, complete provenance/split CSVs, and calibration/firmware details. The original source Git tree has documentation and metadata but not image datasets or .pt binaries. Do not commit personal video/voice footage without consent.

Please provide:
1. Original `li_stick_navigation_v0.3_combined.pt` with SHA256, training run and class map, or legal access instructions.
2. The 9,780 train / 533 validation / 618 test **images and matching YOLO labels**, or private download links and access approvals. Check the existing dataset registry for subsets.
3. Full `data/li_stick_navigation/provenance.csv`, `data/training/li_stick_v02_exp2b/split_assignment.csv`, and `data/training/li_stick_v03_combined/split_assignment.csv`.
4. Dataset licenses, attribution, annotation guidelines, consent/privacy assurances, and excluded categories.
5. Representative footage: curbs, poles, stairs, cars, motorcycles, potholes, overhead hazards, nighttime, rain, glare, occlusion, traffic, dropped cane, fast motion. Do not confuse camera image location with height or measured depth.
6. Hardware inventory and pinout: exact XIAO ESP32-S3 variant, whether separate ESP32-S3-WROOM is used, exact ultrasonic sensor model/range, IMU model and mounting, Grove Vision AI V2 camera, power/battery, feedback actuator(s), wiring diagram, sample serial packets, and firmware revision.
7. Who can run GPU retraining and provide held-out evaluations and on-device timing.

**Data storage:** Keep large images/video and model binaries in licensed external storage, and point to SHA256-verified release artifacts or manifests. Do not put sensitive footage into a public repository. A GitHub LFS workflow may be used where permitted.

**Acceptance:** only publish candidate weights after successful audit, reproducible training, independent hold-out evaluation, per-class recall and misdetection review, and device-specific benchmarking.
