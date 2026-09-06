# AGM-YOLO Core Modules

PyTorch implementation of the three feature-enhancement modules used in **AGM-YOLO: A Wood Defect Detection Algorithm with Illumination Calibration and Multi-Scale Feature Enhancement**:

- **AIC** — Adaptive Illumination Calibration
- **GCE** — Global Context Enhancement
- **MSA** — Multi-Scale Selective Aggregation

This repository is provided as the code companion for the manuscript and focuses on the proposed modules. The source files depend only on PyTorch and can be integrated into an existing detection framework.

## Repository structure

```text
AGM-modules/
├── modules/
│   ├── aic.py
│   ├── gce.py
│   ├── msa.py
│   └── __init__.py
├── tests/
│   └── test_modules.py
├── .github/workflows/tests.yml
├── example.py
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── TESTING.md
└── LICENSE
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Quick start

```bash
python example.py --device cpu
```

If a CUDA-enabled PyTorch installation is available:

```bash
python example.py --device cuda
```

## AIC: Adaptive Illumination Calibration

AIC generates a channel-wise calibration factor from global spatial statistics and applies it to the input feature map:

```text
z = mean(x, spatial dimensions)
s = sigmoid(conv2(relu(conv1(z))))
y = x × (1 + s)
```

```python
import torch
from modules import AIC

x = torch.randn(2, 64, 80, 80)
aic = AIC(channels=64, reduction=16)
y = aic(x)
```

The output has the same shape as the input.

## GCE: Global Context Enhancement

GCE models channel, horizontal, and vertical context with three multiplicative attention branches:

```text
Mc = channel_mlp(mean(x, height and width))
Mw = sigmoid(width_conv(mean(x, height)))
Mh = sigmoid(height_conv(mean(x, width)))
y = x × Mc × Mw × Mh
```

```python
from modules import GCE

gce = GCE(channels=64, reduction=16)
y = gce(x)
```

## MSA: Multi-Scale Selective Aggregation

MSA aligns adjacent-scale features to the current scale and performs spatially adaptive aggregation:

```text
aligned_i = resize(project_1x1(adjacent_i), current spatial size)
weight_i = sigmoid(gate_1x1(concat(current, aligned_i)))
output = current + sum(weight_i × aligned_i)
```

```python
import torch
from modules import MSA

p3 = torch.randn(2, 64, 80, 80)
p4 = torch.randn(2, 128, 40, 40)
p5 = torch.randn(2, 256, 20, 20)

msa3 = MSA([64, 128])
msa4 = MSA([128, 64, 256])
msa5 = MSA([256, 128])

q3 = msa3([p3, p4])
q4 = msa4([p4, p3, p5])
q5 = msa5([p5, p4])
```

For each MSA branch, the first tensor is the current-scale feature and the remaining tensors are adjacent-scale features. All feature tensors must use the same batch size, device, and data type.

## Integration with YOLO

In the AGM-YOLO architecture, the modules are used as feature-enhancement components within the backbone/neck pipeline before the detection head. When integrating them into an Ultralytics-based project, register the custom modules in the model parser and pass the channel dimensions produced by the selected model scale.

The following implementation settings are used in this repository:

| Component | Setting |
| --- | --- |
| AIC/GCE reduction ratio | 16 by default |
| Intermediate activation | ReLU |
| MSA channel alignment | Independent 1×1 convolution for each adjacent scale |
| MSA spatial alignment | Bilinear interpolation with `align_corners=False` |
| MSA gating | Independent sigmoid spatial gate for each adjacent scale |
| Parameter initialization | PyTorch default initialization |

## Tests

The test suite checks tensor shapes, mathematical operations, gradients, serialization, error handling, non-square feature maps, and optional CUDA mixed-precision execution.

```bash
python -m pytest
```

See [TESTING.md](TESTING.md) for additional information.


## Citation

If this code is useful in your research, please cite the accompanying AGM-YOLO manuscript. Publication metadata will be added here after the paper is formally published.

## License

The code in this repository is released under the MIT License. See [LICENSE](LICENSE).
