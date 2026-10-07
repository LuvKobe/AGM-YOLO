# Reproducibility protocol

This repository provides a fixed implementation and command-line workflow for AGM-YOLO experiments.

## Software configuration

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1 |
| Ultralytics | 8.3.138 |
| OpenCV | 4.10.0.84 |
| NumPy | 1.26.4 |
| PyYAML | 6.0.2 |
| pandas | 2.2.3 |
| pycocotools | 2.0.8 |

CUDA 12.1 is used for GPU training and evaluation.

## Golden-Y training protocol

- Model: AGM-YOLO11n
- Input size: 640 × 640
- Optimizer: AdamW
- Epochs: 200
- Batch size: 8
- Initial learning rate: 0.01
- Final learning-rate ratio: 0.01
- Weight decay: 0.0005
- Warm-up epochs: 3
- Warm-up momentum: 0.8
- Warm-up bias learning rate: 0.1
- AMP: enabled
- Deterministic execution: enabled

Main command:

```bash
python -m scripts.train \
  --model models/agm-yolo11n.yaml \
  --data /path/to/golden_y/data.yaml \
  --pretrained yolo11n.pt \
  --device 0 \
  --seed 0
```

## Factorial ablation protocol

The repository includes all combinations of the three proposed components:

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

Run:

```bash
python -m scripts.run_ablation --data /path/to/golden_y/data.yaml --device 0 --seed 0
```

## Repeated-run protocol

```bash
python -m scripts.train_multiseed \
  --model models/agm-yolo11n.yaml \
  --data /path/to/golden_y/data.yaml \
  --seeds 0 1 2 \
  --device 0
```

## Scale evaluation

AP_S, AP_M and AP_L use COCO area thresholds on the 640 × 640 letterboxed evaluation canvas.

```bash
python -m experiments.scale_ap \
  --weights best.pt \
  --data /path/to/golden_y/data.yaml \
  --imgsz 640 \
  --device 0
```

## Robustness evaluation

Brightness factors:

```text
1.0, 0.8, 0.6, 0.4
```

Gaussian noise standard deviations at brightness factor 0.6:

```text
0.01, 0.02
```

Run:

```bash
python -m experiments.run_stress_suite \
  --weights best.pt \
  --data /path/to/golden_y/data.yaml \
  --device 0 \
  --fp-conf 0.25
```

`FP/image` uses class-aware greedy matching with IoU ≥ 0.50 and the fixed score threshold specified by `--fp-conf`.

## Low-light evaluation

Images are ranked using normalized Rec.709 luminance, and the lowest luminance quartile is evaluated as the low-light subset.

```bash
python -m experiments.lowlight \
  --weights best.pt \
  --data /path/to/golden_y/data.yaml \
  --device 0 \
  --fp-conf 0.25
```

## GPU benchmark

```bash
python -m experiments.benchmark \
  --weights best.pt \
  --device cuda:0 \
  --imgsz 640 \
  --warmup 50 \
  --runs 300 \
  --half
```

Timing uses synchronized CUDA execution with batch size 1 and model-forward latency.

## VSB protocol

The VSB evaluation uses a deterministic 2,800-image subset generated from sorted source paths with seed 42. The converter maps the retained seven classes to YOLO format and writes a split manifest.

```bash
python -m tools.prepare_vsb \
  --images /path/to/VSB/Images \
  --boxes /path/to/VSB/Bounding_Boxes \
  --out datasets/vsb_7class_seed42 \
  --subset-size 2800 \
  --seed 42 \
  --test-fraction 0.30 \
  --val-fraction-of-train 0.10 \
  --resize 1280 512
```

Training:

```bash
python -m scripts.train_vsb \
  --data datasets/vsb_7class_seed42/data.yaml \
  --device 0 \
  --seed 42
```

## Output records

Training and evaluation utilities save machine-readable configuration and metric files alongside standard Ultralytics outputs. These records include run parameters, environment information, metrics, split manifests, and benchmark results where applicable.
