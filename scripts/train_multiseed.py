"""Run the same training protocol with multiple independent seeds."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="models/agm-yolo11n.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--pretrained", default="yolo11n.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--project", default="runs/multiseed")
    args = p.parse_args()
    for seed in args.seeds:
        cmd = [
            sys.executable, "-m", "scripts.train",
            "--model", args.model,
            "--data", args.data,
            "--pretrained", args.pretrained,
            "--device", str(args.device),
            "--seed", str(seed),
            "--project", args.project,
            "--name", f"seed{seed}",
        ]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
