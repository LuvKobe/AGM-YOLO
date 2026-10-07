"""Helpers to construct or load AGM-YOLO models."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .integration import patch_ultralytics


def build_model(
    cfg: str | Path = "models/agm-yolo11n.yaml",
    pretrained: Optional[str | Path] = "yolo11n.pt",
):
    patch_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(cfg), task="detect")
    if pretrained:
        model.load(str(pretrained))
    return model


def load_model(weights: str | Path):
    patch_ultralytics()
    from ultralytics import YOLO

    return YOLO(str(weights), task="detect")
