"""Evaluate the lowest-luminance quartile of the untouched test split."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from agm_yolo.data import class_names, iter_split_images, label_path_for_image, load_data_yaml
from agm_yolo.model import load_model
from experiments.robustness import extract_metrics, false_positives


def mean_luminance(path: Path) -> float:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return float(y.mean())


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--fp-conf", type=float, default=0.25)
    p.add_argument("--fp-iou", type=float, default=0.50)
    p.add_argument("--cache", default="runs/lowlight_subset")
    p.add_argument("--out", default="results/lowlight.json")
    args = p.parse_args()

    data = load_data_yaml(args.data)
    names = class_names(data)
    images = iter_split_images(args.data, "test")
    ranked = sorted(((mean_luminance(p), p) for p in images), key=lambda x: (x[0], str(x[1])))
    n = max(1, int(np.floor(len(images) * 0.25)))
    selected = ranked[:n]

    out_dir = Path(args.cache)
    if out_dir.exists(): shutil.rmtree(out_dir)
    (out_dir / "images/test").mkdir(parents=True)
    (out_dir / "labels/test").mkdir(parents=True)
    records = []
    manifest = []
    for i, (lum, img) in enumerate(selected):
        dst_img = out_dir / "images/test" / f"{i:04d}_{img.name}"
        dst_lab = out_dir / "labels/test" / f"{i:04d}_{img.stem}.txt"
        try: dst_img.hardlink_to(img.resolve())
        except OSError: shutil.copy2(img, dst_img)
        src_lab = label_path_for_image(img)
        if src_lab.exists(): shutil.copy2(src_lab, dst_lab)
        else: dst_lab.write_text("", encoding="utf-8")
        records.append((dst_img, src_lab))
        manifest.append({"source": str(img), "mean_luminance": lum})

    y = out_dir / "data.yaml"
    y.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()), "train": "images/test", "val": "images/test", "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False), encoding="utf-8")
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    model = load_model(args.weights)
    metrics = model.val(data=str(y), split="test", device=args.device, imgsz=args.imgsz,
                        batch=args.batch, conf=0.001, iou=0.7, max_det=300, plots=False,
                        project="runs/lowlight_val", name=Path(args.weights).stem)
    predictions = model.predict(source=[str(r[0]) for r in records], imgsz=args.imgsz,
                                device=args.device, conf=args.fp_conf, iou=0.7,
                                max_det=300, verbose=False)
    fp, tp, gt = false_positives(predictions, records, args.fp_iou)
    payload = {
        "weights": args.weights, "images": len(records), **extract_metrics(metrics),
        "fp_conf": args.fp_conf, "fp_match_iou": args.fp_iou,
        "fp_per_image": fp / max(1, len(records)), "false_positives": fp,
        "tp_at_fixed_threshold": tp, "gt_boxes": gt,
        "selection": "lowest floor(N*0.25) images by mean Rec.709 normalized RGB luminance",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
