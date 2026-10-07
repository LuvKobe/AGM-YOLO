"""COCO AP_S/AP_M/AP_L on a 640x640 letterbox evaluation canvas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from agm_yolo.data import class_names, iter_split_images, label_path_for_image, load_data_yaml, read_yolo_labels
from agm_yolo.model import load_model


def letterbox_geometry(w: int, h: int, canvas: int):
    gain = min(canvas / w, canvas / h)
    new_w, new_h = w * gain, h * gain
    pad_x, pad_y = (canvas - new_w) / 2.0, (canvas - new_h) / 2.0
    return gain, pad_x, pad_y


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="results/scale_ap.json")
    args = p.parse_args()

    data = load_data_yaml(args.data)
    names = class_names(data)
    images = iter_split_images(args.data, args.split)
    if not images:
        raise SystemExit("No images found")

    coco_gt = {
        "info": {"description": "AGM-YOLO scale-specific evaluation"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
    }
    metadata = []
    ann_id = 1
    for image_id, image_path in enumerate(images, 1):
        im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(image_path)
        h, w = im.shape[:2]
        gain, pad_x, pad_y = letterbox_geometry(w, h, args.imgsz)
        coco_gt["images"].append({"id": image_id, "file_name": image_path.name, "width": args.imgsz, "height": args.imgsz})
        for c, xc, yc, bw, bh in read_yolo_labels(label_path_for_image(image_path)):
            x1 = (xc - bw / 2) * w * gain + pad_x
            y1 = (yc - bh / 2) * h * gain + pad_y
            ww, hh = bw * w * gain, bh * h * gain
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": c + 1,
                "bbox": [x1, y1, ww, hh],
                "area": ww * hh,
                "iscrowd": 0,
            })
            ann_id += 1
        metadata.append((image_id, image_path, w, h, gain, pad_x, pad_y))

    model = load_model(args.weights)
    preds = model.predict(
        source=[str(p) for p in images], imgsz=args.imgsz, device=args.device,
        conf=0.001, iou=0.7, max_det=300, verbose=False, stream=False,
    )
    coco_pred = []
    for result, (image_id, _path, _w, _h, gain, pad_x, pad_y) in zip(preds, metadata):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, c in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = box.tolist()
            x1, x2 = x1 * gain + pad_x, x2 * gain + pad_x
            y1, y2 = y1 * gain + pad_y, y2 * gain + pad_y
            coco_pred.append({
                "image_id": image_id,
                "category_id": int(c) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    gt = COCO(); gt.dataset = coco_gt; gt.createIndex()
    dt = gt.loadRes(coco_pred) if coco_pred else gt.loadRes([])
    ev = COCOeval(gt, dt, "bbox")
    # pycocotools defaults: small < 32^2, medium 32^2..96^2, large > 96^2.
    ev.params.imgIds = [x[0] for x in metadata]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    payload = {
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "canvas": [args.imgsz, args.imgsz],
        "AP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "AP_S": float(ev.stats[3]),
        "AP_M": float(ev.stats[4]),
        "AP_L": float(ev.stats[5]),
        "ground_truth_boxes": len(coco_gt["annotations"]),
        "predictions": len(coco_pred),
        "area_thresholds_pixels2": {"small_max": 32 ** 2, "medium_max": 96 ** 2},
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
