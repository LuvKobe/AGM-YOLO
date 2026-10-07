# AGM-YOLO

PyTorch/Ultralytics implementation of **AGM-YOLO: A Wood Defect Detection Algorithm with Illumination Calibration and Multi-Scale Feature Enhancement**.

AGM-YOLO is built on YOLO11n and introduces three task-oriented components for wood-surface defect detection:

- **AIC (Adaptive Illumination Calibration):** channel-wise residual recalibration for illumination-sensitive features;
- **GCE (Global Context Enhancement):** joint channel, horizontal, and vertical context modeling;
- **MSA (Multi-Scale Selective Aggregation):** gated adjacent-scale feature fusion before the detection head.

This repository provides the model implementation, complete AIC/GCE/MSA ablation configurations, training and validation scripts, dataset preparation utilities, robustness evaluation, scale-specific AP evaluation, latency/memory benchmarking, TensorRT export, and visualization tools.

## 1. Repository structure

```text
AGM-YOLO/
├── agm_yolo/                  # Core modules and Ultralytics integration
├── models/
│   ├── agm-yolo11n.yaml       # Full AGM-YOLO model
│   ├── yolo11n-baseline.yaml
│   └── ablations/             # Eight AIC/GCE/MSA combinations
├── scripts/                   # Train, validate, predict and ablation runs
├── tools/                     # Dataset preparation and split utilities
├── experiments/               # Robustness, scale AP, benchmarking, Grad-CAM
├── tests/                     # Unit and integration tests
├── environment.yml
├── requirements.txt
└── REPRODUCIBILITY.md
```

## 2. Environment

The experiments use:

- Python 3.10
- PyTorch 2.5.1
- CUDA 12.1
- Ultralytics 8.3.138
- mixed-precision training

Recommended installation:

```bash
conda create -n agm-yolo python=3.10 -y
conda activate agm-yolo

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps
```

Run the tests:

```bash
pytest
python -m scripts.smoke_test
```

The smoke test reports AIC at backbone layers `[2, 4, 6, 8]`, GCE at layer `[10]`, and the MSA-enabled detection head at layer `[23]`.

## 3. Model configuration

The full model is defined in:

```text
models/agm-yolo11n.yaml
```

AGM-YOLO preserves the standard YOLO11n stage layout. AIC is attached to the backbone C3k2 stages, GCE follows C2PSA, and MSA operates on P3/P4/P5 immediately before detection.

The implementation is integrated with Ultralytics at runtime through `agm_yolo/integration.py`, so no manual modification of the installed Ultralytics package is required.

## 4. Golden-Y dataset

Primary dataset:

**Wood Defect Detection Dataset v1, Golden Y, Roboflow Universe**  
<https://universe.roboflow.com/yolo11-yqwwf/wood-defect-detection-dataset-w0ws3/dataset/1>

Dataset split:

| Split | Images |
|---|---:|
| Train | 2,641 |
| Validation | 755 |
| Test | 377 |

Class order:

```text
0 dry knot
1 edge knot
2 small knot
3 sound knot
4 split
5 wave
```

After exporting the dataset in Ultralytics YOLO format, the split can be summarized with:

```bash
python -m tools.audit_golden_y \
  --data D:/datasets/wood-defect/data.yaml \
  --write-manifest datasets/golden_y_manifest.csv \
  --strict
```

## 5. Training

### AGM-YOLO

```bash
python -m scripts.train \
  --model models/agm-yolo11n.yaml \
  --data D:/datasets/wood-defect/data.yaml \
  --pretrained yolo11n.pt \
  --device 0 \
  --seed 0 \
  --project runs/golden_y \
  --name agm_yolo_seed0
```

### YOLO11n baseline

```bash
python -m scripts.train \
  --model models/ablations/baseline.yaml \
  --data D:/datasets/wood-defect/data.yaml \
  --pretrained yolo11n.pt \
  --device 0 \
  --seed 0 \
  --project runs/golden_y \
  --name yolo11n_seed0
```

Default training settings are AdamW, 200 epochs, 3 warm-up epochs, batch size 8, image size 640, `lr0=0.01`, `lrf=0.01`, weight decay `5e-4`, AMP, and deterministic execution.

## 6. Validation

```bash
python -m scripts.val \
  --weights runs/golden_y/agm_yolo_seed0/weights/best.pt \
  --data D:/datasets/wood-defect/data.yaml \
  --split test \
  --device 0
```

Evaluation artifacts and machine-readable metrics are written to the run directory.

## 7. AIC/GCE/MSA ablation study

Eight factorial configurations are included:

```text
baseline
aic
gce
msa
aic+gce
aic+msa
gce+msa
aic+gce+msa
```

Run the full ablation study:

```bash
python -m scripts.run_ablation \
  --data D:/datasets/wood-defect/data.yaml \
  --device 0 \
  --seed 0
```

The YAML files can be regenerated with:

```bash
python -m scripts.generate_ablation_yamls
```

## 8. Repeated runs

Independent seeds can be launched with:

```bash
python -m scripts.train_multiseed \
  --model models/agm-yolo11n.yaml \
  --data D:/datasets/wood-defect/data.yaml \
  --seeds 0 1 2 \
  --device 0
```

## 9. Scale-specific AP

Small, medium, and large object AP are evaluated on the 640×640 letterboxed canvas using COCO area thresholds:

- small: area < 32² px²
- medium: 32² ≤ area < 96² px²
- large: area ≥ 96² px²

```bash
python -m experiments.scale_ap \
  --weights runs/golden_y/agm_yolo_seed0/weights/best.pt \
  --data D:/datasets/wood-defect/data.yaml \
  --split test \
  --imgsz 640 \
  --device 0 \
  --out results/agm_scale_ap.json
```

## 10. Illumination and noise robustness

The stress suite evaluates brightness scaling with `alpha ∈ {1.0, 0.8, 0.6, 0.4}` and Gaussian noise at the `alpha=0.6` condition with `sigma ∈ {0.01, 0.02}`.

```bash
python -m experiments.run_stress_suite \
  --weights runs/golden_y/agm_yolo_seed0/weights/best.pt \
  --data D:/datasets/wood-defect/data.yaml \
  --device 0 \
  --fp-conf 0.25 \
  --out results/agm_stress.json
```

`FP/image` is computed by class-aware greedy matching at IoU ≥ 0.50 with the fixed confidence threshold supplied by `--fp-conf`.

## 11. Low-light subset evaluation

The low-light evaluation ranks test images by normalized Rec.709 luminance and evaluates the lowest luminance quartile.

```bash
python -m experiments.lowlight \
  --weights runs/golden_y/agm_yolo_seed0/weights/best.pt \
  --data D:/datasets/wood-defect/data.yaml \
  --device 0 \
  --fp-conf 0.25 \
  --out results/agm_lowlight.json
```

## 12. Latency and memory benchmark

For CUDA FP16 inference:

```bash
python -m experiments.benchmark \
  --weights runs/golden_y/agm_yolo_seed0/weights/best.pt \
  --device cuda:0 \
  --imgsz 640 \
  --warmup 50 \
  --runs 300 \
  --half \
  --out results/agm_fp16.json
```

The benchmark uses batch size 1, synchronized CUDA timing, warm-up iterations, and peak CUDA memory measurement.

## 13. TensorRT export

```bash
python -m experiments.export_tensorrt \
  --weights best.pt \
  --imgsz 640 \
  --device 0
```

The exported TensorRT engine can be benchmarked on the target deployment platform using the same input resolution and batch size.

## 14. VSB dataset evaluation

VSB wood surface defect dataset:

- Zenodo: <https://doi.org/10.5281/zenodo.4694695>
- Dataset paper: <https://doi.org/10.12688/f1000research.52903.2>

Seven retained classes:

```text
0 Live Knot
1 Dead Knot
2 Marrow
3 Resin Pocket
4 Knot with Crack
5 Knot Missing
6 Crack
```

Prepare the 2,800-image seed-42 split:

```bash
python -m tools.prepare_vsb \
  --images D:/datasets/VSB/Images \
  --boxes D:/datasets/VSB/Bounding_Boxes \
  --out datasets/vsb_7class_seed42 \
  --subset-size 2800 \
  --seed 42 \
  --test-fraction 0.30 \
  --val-fraction-of-train 0.10 \
  --resize 1280 512
```

Train on the prepared dataset:

```bash
python -m scripts.train_vsb \
  --data datasets/vsb_7class_seed42/data.yaml \
  --device 0 \
  --seed 42
```

The conversion utility writes `data.yaml`, `split_manifest.csv`, and `conversion_report.json` together with the converted YOLO annotations.

## 15. Prediction and Grad-CAM

Prediction:

```bash
python -m scripts.predict \
  --weights best.pt \
  --source path/to/test_images \
  --conf 0.25 \
  --device 0
```

Grad-CAM visualization:

```bash
python -m experiments.gradcam \
  --weights best.pt \
  --image path/to/test_image.jpg \
  --layer 22 \
  --device 0 \
  --out results/gradcam.png
```

## 16. Reproducibility

A compact experiment protocol is provided in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Training runs also save configuration and environment metadata to support consistent repeated evaluation.

## 17. License

The AGM-YOLO source code in this repository is released under the MIT License. Ultralytics and the referenced datasets remain subject to their respective licenses and terms.

## 18. Citation

```bibtex
@article{agm_yolo_2026,
  title   = {AGM-YOLO: A Wood Defect Detection Algorithm with Illumination Calibration and Multi-Scale Feature Enhancement},
  author  = {Hu, Cheng and Chen, Yajun and Liu, Wenhao},
  year    = {2026}
}
```
