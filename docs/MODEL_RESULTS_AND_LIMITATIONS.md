# Model results and release gates

- Original baseline from `MODELS.md`: combined v0.3 mAP@50 **0.331**, mAP@50:95 **0.215**, evaluated in the original project.
- **New model accuracy: unavailable.** No v0.4 or v0.5 trained checkpoint exists in this repository.
- The original `models/li_stick_navigation_v0.3_combined.pt` is *not* in the source Git tree.
- No public dataset images or private team images are included. `config/dataset_v04.yaml` is a **template**, not working data.
- Software tests do not measure real-world navigation safety. Candidate thresholds must be calibrated and verified.
- Check dataset reuse licensing, frame/video split leakage and domain shift before retraining.
- Once images and weights are provided: run dataset audit, training, held-out evaluation, per-class error analysis, device-specific profiling and supervised usability studies.
- Potholes/drop-offs/overhead objects need new classes and labels; existing 13-class checkpoint does not support them natively.

## Commands
```bash
python -m training_v04.audit --data config/dataset_v04.yaml
python -m training_v04.train --data config/dataset_v04.yaml --weights models/li_stick_navigation_v0.3_combined.pt --epochs 100 --device 0
python -m training_v04.evaluate --data config/dataset_v04.yaml --weights outputs/train_v04/experiment/weights/best.pt
streamlit run ui/dashboard.py
```

## Important deployment caveat
A `.pt` file or an ONNX export alone is not an executable model for the Himax processor on Grove Vision AI V2. Verify supported architectures, quantization, memory constraints, TensorRT/PyTorch-to-Himax pipeline requirements and runtime latency against the exact module revision before promising on-device performance.
