"""Run the AGM-YOLO brightness/noise grid for one checkpoint."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--fp-conf", default="0.25")
    p.add_argument("--out", default="results/stress.json")
    args = p.parse_args()
    conditions = [(1.0, 0.0), (0.8, 0.0), (0.6, 0.0), (0.4, 0.0), (0.6, 0.01), (0.6, 0.02)]
    for alpha, sigma in conditions:
        cmd = [sys.executable, "-m", "experiments.robustness",
               "--weights", args.weights, "--data", args.data, "--device", args.device,
               "--alpha", str(alpha), "--sigma", str(sigma), "--fp-conf", str(args.fp_conf),
               "--out", args.out]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
