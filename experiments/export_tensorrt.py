"""Export a trained checkpoint to TensorRT FP16 for Jetson/desktop benchmarking."""
from __future__ import annotations

import argparse
from agm_yolo.model import load_model


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace GiB")
    args = p.parse_args()
    model = load_model(args.weights)
    exported = model.export(format="engine", imgsz=args.imgsz, batch=1, half=True,
                            dynamic=False, device=args.device, workspace=args.workspace)
    print(exported)


if __name__ == "__main__":
    main()
