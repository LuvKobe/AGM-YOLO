"""Brightness and Gaussian-noise stress evaluation for the unchanged test set."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from agm_yolo.data import class_names, iter_split_images, label_path_for_image, load_data_yaml, read_yolo_labels
from agm_yolo.model import load_model


def transform_image(image: np.ndarray, alpha: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    x *= alpha
    if sigma > 0:
        x += rng.normal(0.0, sigma, x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0) * 255.0


def xywh_to_xyxy(box):
    _, xc, yc, w, h = box
    return np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dtype=float)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def false_positives(results, source_records, match_iou: float) -> tuple[int, int, int]:
    fp = tp = gt_total = 0
    for result, (_, src_label) in zip(results, source_records):
        gts = read_yolo_labels(src_label)
        gt_total += len(gts)
        gt_boxes = [(int(c), xywh_to_xyxy(row)) for row in gts for c in [row[0]]]
        matched = set()
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        pred_cls = boxes.cls.detach().cpu().numpy().astype(int)
        pred_conf = boxes.conf.detach().cpu().numpy()
        pred_xywhn = boxes.xywhn.detach().cpu().numpy()
        order = np.argsort(-pred_conf)
        for j in order:
            c = pred_cls[j]
            xc, yc, w, h = pred_xywhn[j]
            pbox = np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dtype=float)
            candidates = [(k, iou_xyxy(pbox, gbox)) for k, (gc, gbox) in enumerate(gt_boxes)
                          if gc == c and k not in matched]
            if candidates:
                k, best = max(candidates, key=lambda x: x[1])
                if best >= match_iou:
                    matched.add(k); tp += 1; continue
            fp += 1
    return fp, tp, gt_total


def materialize_stress_set(data_yaml: str, out: Path, alpha: float, sigma: float, seed: int):
    data = load_data_yaml(data_yaml)
    names = class_names(data)
    images = iter_split_images(data_yaml, "test")
    if out.exists():
        shutil.rmtree(out)
    (out / "images/test").mkdir(parents=True)
    (out / "labels/test").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    records = []
    for i, image_path in enumerate(images):
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(image_path)
        dst_img = out / "images/test" / f"{i:05d}_{image_path.stem}.png"
        dst_lab = out / "labels/test" / f"{i:05d}_{image_path.stem}.txt"
        transformed = transform_image(raw, alpha, sigma, rng).round().astype(np.uint8)
        cv2.imwrite(str(dst_img), transformed)
        src_lab = label_path_for_image(image_path)
        if src_lab.exists():
            shutil.copy2(src_lab, dst_lab)
        else:
            dst_lab.write_text("", encoding="utf-8")
        records.append((dst_img, src_lab))
    generated = {
        "path": str(out.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    y = out / "data.yaml"
    y.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    return y, records


def extract_metrics(metrics) -> dict:
    d = dict(getattr(metrics, "results_dict", {}) or {})
    def find(*keys):
        for key in keys:
            if key in d:
                return float(d[key])
        return None
    return {
        "precision": find("metrics/precision(B)", "metrics/precision"),
        "recall": find("metrics/recall(B)", "metrics/recall"),
        "map50": find("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": find("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="Noise RNG seed; ignored when sigma=0")
    p.add_argument("--fp-conf", type=float, default=0.25,
                   help="Fixed score threshold used for FP/image")
    p.add_argument("--fp-iou", type=float, default=0.50, help="Class-aware GT matching IoU for FP/image")
    p.add_argument("--cache-root", default="runs/stress_cache")
    p.add_argument("--out", default="results/stress.json")
    args = p.parse_args()

    tag = f"a{args.alpha:.2f}_s{args.sigma:.3f}_seed{args.seed}".replace(".", "p")
    stress_dir = Path(args.cache_root) / tag
    generated_yaml, records = materialize_stress_set(args.data, stress_dir, args.alpha, args.sigma, args.seed)
    model = load_model(args.weights)
    metrics = model.val(
        data=str(generated_yaml), split="test", device=args.device, imgsz=args.imgsz,
        batch=args.batch, conf=0.001, iou=0.7, max_det=300, plots=False,
        project="runs/stress_val", name=tag,
    )
    predictions = model.predict(
        source=[str(r[0]) for r in records], imgsz=args.imgsz, device=args.device,
        conf=args.fp_conf, iou=0.7, max_det=300, verbose=False, stream=False,
    )
    fp, tp_at_fixed, gt_total = false_positives(predictions, records, args.fp_iou)
    payload = {
        "weights": args.weights,
        "alpha": args.alpha,
        "sigma": args.sigma,
        "images": len(records),
        **extract_metrics(metrics),
        "fp_conf": args.fp_conf,
        "fp_match_iou": args.fp_iou,
        "false_positives": fp,
        "fp_per_image": fp / max(1, len(records)),
        "tp_at_fixed_threshold": tp_at_fixed,
        "gt_boxes": gt_total,
        "note": "mAP/P/R use Ultralytics validation; FP/image uses class-aware greedy matching at the explicitly recorded fixed confidence threshold.",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try: existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception: existing = []
        if not isinstance(existing, list): existing = [existing]
    else:
        existing = []
    existing.append(payload)
    out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
