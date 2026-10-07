"""Train all eight AIC/GCE/MSA ablations with the AGM-YOLO protocol."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pretrained", default="yolo11n.pt")
    p.add_argument("--project", default="runs/ablation")
    args = p.parse_args()
    for cfg in sorted(Path("models/ablations").glob("*.yaml")):
        cmd = [
            sys.executable, "-m", "scripts.train",
            "--model", str(cfg),
            "--data", args.data,
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--pretrained", args.pretrained,
            "--project", args.project,
            "--name", cfg.stem,
        ]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
