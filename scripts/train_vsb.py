"""Train AGM-YOLO on a prepared VSB split using the configured optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agm_yolo.model import build_model
from agm_yolo.repro import environment_snapshot, seed_everything


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml produced by tools.prepare_vsb")
    p.add_argument("--model", default="models/agm-yolo11n.yaml")
    p.add_argument("--pretrained", default="yolo11n.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Longest side; use --rect for native ~1280x512 batches")
    p.add_argument("--project", default="runs/vsb")
    p.add_argument("--name", default="agm_yolo_vsb_seed42")
    args = p.parse_args()

    seed_everything(args.seed)
    model = build_model(args.model, None if args.pretrained.lower() == "none" else args.pretrained)
    settings = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        rect=True,
        batch=args.batch,
        device=args.device,
        optimizer="SGD",
        lr0=5e-4,
        lrf=0.01,
        momentum=0.9,
        cos_lr=False,
        seed=args.seed,
        deterministic=True,
        amp=True,
        project=args.project,
        name=args.name,
        plots=True,
        patience=0,
    )
    model.train(**settings)
    save_dir = Path(model.trainer.save_dir)
    (save_dir / "reproduction_config.json").write_text(
        json.dumps({"train": settings, "environment": environment_snapshot()}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
