"""Train AGM-YOLO or any generated ablation model on a YOLO dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agm_yolo.model import build_model
from agm_yolo.repro import environment_snapshot, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="models/agm-yolo11n.yaml")
    p.add_argument("--data", required=True, help="Path to Ultralytics data.yaml")
    p.add_argument("--pretrained", default="yolo11n.pt", help="Set to 'none' for random init")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/golden_y")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    pretrained = None if str(args.pretrained).lower() in {"none", "false", "0"} else args.pretrained
    model = build_model(args.model, pretrained=pretrained)
    name = args.name or f"{Path(args.model).stem}_seed{args.seed}"
    settings = dict(
        data=str(Path(args.data).expanduser()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        amp=True,
        workers=args.workers,
        cache=args.cache,
        project=args.project,
        name=name,
        exist_ok=args.exist_ok,
        plots=True,
        patience=0,
        resume=args.resume,
    )
    model.train(**settings)
    save_dir = Path(model.trainer.save_dir)
    (save_dir / "reproduction_config.json").write_text(
        json.dumps({"train": settings, "environment": environment_snapshot()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Run saved to: {save_dir}")


if __name__ == "__main__":
    main()
