"""Run AGM-YOLO inference and save annotated images."""
from __future__ import annotations

import argparse
from agm_yolo.model import load_model


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--project", default="runs/predict")
    p.add_argument("--name", default="agm_yolo")
    args = p.parse_args()
    model = load_model(args.weights)
    model.predict(source=args.source, device=args.device, imgsz=args.imgsz, conf=args.conf,
                  iou=args.iou, save=True, save_txt=True, save_conf=True,
                  project=args.project, name=args.name)


if __name__ == "__main__":
    main()
