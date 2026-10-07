"""Evaluate a trained AGM-YOLO checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agm_yolo.model import load_model
from agm_yolo.repro import environment_snapshot


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--project", default="runs/val")
    p.add_argument("--name", default="agm_yolo")
    args = p.parse_args()

    model = load_model(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=300,
        plots=True,
        project=args.project,
        name=args.name,
    )
    save_dir = Path(metrics.save_dir)
    payload = {
        "metrics": getattr(metrics, "results_dict", {}),
        "environment": environment_snapshot(),
        "weights": str(Path(args.weights).expanduser()),
        "data": str(Path(args.data).expanduser()),
        "split": args.split,
    }
    (save_dir / "reproduction_metrics.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
