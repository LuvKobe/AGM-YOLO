from __future__ import annotations

from pathlib import Path
import yaml


def load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def toggles(cfg: dict) -> tuple[bool, bool, bool]:
    layers = cfg["backbone"] + cfg["head"]
    aic = bool(layers[2][3][5])
    gce = bool(layers[10][3][2])
    msa = bool(layers[23][3][1])
    return aic, gce, msa


def test_full_architecture_indices_and_toggles():
    cfg = load(Path("models/agm-yolo11n.yaml"))
    layers = cfg["backbone"] + cfg["head"]
    assert cfg["scale"] == "n"
    assert len(layers) == 24
    assert toggles(cfg) == (True, True, True)
    assert layers[23][0] == [16, 19, 22]
    assert layers[23][2] == "Detect"


def test_all_eight_ablation_combinations_exist():
    configs = sorted(Path("models/ablations").glob("*.yaml"))
    assert len(configs) == 8
    found = {toggles(load(path)) for path in configs}
    expected = {(a, g, m) for a in (False, True) for g in (False, True) for m in (False, True)}
    assert found == expected
