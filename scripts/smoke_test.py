"""Build AGM-YOLO from YAML, run one forward pass, and print integration status."""
from __future__ import annotations

import json
import torch
from torch import nn

from agm_yolo.integration import C2PSAAGM, C3k2AGM, DetectAGM, patch_ultralytics
from agm_yolo.repro import environment_snapshot


def main() -> None:
    patch_ultralytics(strict_version=True)
    from ultralytics import YOLO

    wrapper = YOLO("models/agm-yolo11n.yaml", task="detect")
    layers = wrapper.model.model
    status = {
        "AIC_layers": [i for i, m in enumerate(layers) if isinstance(m, C3k2AGM) and not isinstance(m.aic, nn.Identity)],
        "GCE_layers": [i for i, m in enumerate(layers) if isinstance(m, C2PSAAGM) and not isinstance(m.gce, nn.Identity)],
        "MSA_detect_layer": [i for i, m in enumerate(layers) if isinstance(m, DetectAGM) and m.use_msa],
        "parameters": sum(p.numel() for p in wrapper.model.parameters()),
        "environment": environment_snapshot(),
    }
    wrapper.model.eval()
    with torch.no_grad():
        out = wrapper.model(torch.zeros(1, 3, 640, 640))
    pred = out[0] if isinstance(out, tuple) else out
    status["forward_output_shape"] = list(pred.shape) if hasattr(pred, "shape") else str(type(pred))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
