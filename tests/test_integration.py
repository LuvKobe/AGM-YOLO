from __future__ import annotations

import pytest


def test_agm_yaml_builds():
    pytest.importorskip("ultralytics")
    import torch
    from torch import nn
    from agm_yolo.integration import C2PSAAGM, C3k2AGM, DetectAGM, patch_ultralytics
    from ultralytics import YOLO

    patch_ultralytics(strict_version=True)
    model = YOLO("models/agm-yolo11n.yaml", task="detect")
    layers = list(model.model.model)
    assert isinstance(layers[2], C3k2AGM) and not isinstance(layers[2].aic, nn.Identity)
    assert isinstance(layers[4], C3k2AGM) and not isinstance(layers[4].aic, nn.Identity)
    assert isinstance(layers[6], C3k2AGM) and not isinstance(layers[6].aic, nn.Identity)
    assert isinstance(layers[8], C3k2AGM) and not isinstance(layers[8].aic, nn.Identity)
    assert isinstance(layers[10], C2PSAAGM) and not isinstance(layers[10].gce, nn.Identity)
    assert isinstance(layers[23], DetectAGM) and layers[23].use_msa

    # Smoke forward verifies MSA branch shapes and parser wiring.
    model.model.eval()
    with torch.no_grad():
        _ = model.model(torch.zeros(1, 3, 640, 640))
