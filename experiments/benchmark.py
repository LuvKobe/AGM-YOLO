"""Batch-1 synchronized CUDA latency and peak-memory benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from agm_yolo.model import load_model
from agm_yolo.repro import environment_snapshot


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--half", action="store_true", help="Use FP16 model/input")
    p.add_argument("--out", default="results/benchmark.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")
    device = torch.device(args.device)
    wrapper = load_model(args.weights)
    net = wrapper.model.to(device).eval()
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    if args.half:
        net = net.half(); x = x.half()
    else:
        net = net.float(); x = x.float()

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = net(x)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.runs):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = net(x)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=float)
    mean_ms = float(arr.mean())
    payload = {
        "weights": str(Path(args.weights)),
        "batch": 1,
        "imgsz": args.imgsz,
        "precision": "FP16" if args.half else "FP32",
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "latency_ms_mean": mean_ms,
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_std": float(statistics.pstdev(times)),
        "fps_from_mean_latency": 1000.0 / mean_ms,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "environment": environment_snapshot(),
        "protocol": "batch=1; excludes data loading/visualization; CUDA sync before/after each timed forward",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
