"""Audit a Roboflow/Ultralytics Golden-Y export and freeze split filenames."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from agm_yolo.data import class_names, iter_split_images, label_path_for_image, load_data_yaml, read_yolo_labels, resolve_dataset_root

EXPECTED_COUNTS = {"train": 2641, "val": 755, "test": 377}
EXPECTED_NAMES = ["dry knot", "edge knot", "small knot", "sound knot", "split", "wave"]


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def normalize_name(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True, help="Golden-Y data.yaml from the downloaded YOLO export")
    p.add_argument("--write-manifest", default="datasets/golden_y_manifest.csv")
    p.add_argument("--hash-images", action="store_true", help="SHA256 every image and store the hash in the manifest")
    p.add_argument("--strict", action="store_true", help="Fail if the expected dataset counts or class names do not match")
    args = p.parse_args()

    data = load_data_yaml(args.data)
    names = class_names(data)
    dataset_root = resolve_dataset_root(data)
    normalized = [normalize_name(n) for n in names]
    expected = [normalize_name(n) for n in EXPECTED_NAMES]
    print("classes:", names)
    if normalized != expected:
        msg = f"Class order differs from the expected configuration. expected={EXPECTED_NAMES}, got={names}"
        if args.strict:
            raise SystemExit(msg)
        print("WARNING:", msg)

    manifest = Path(args.write_manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    total_boxes = 0
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "label", "boxes", "image_sha256"])
        for split in ("train", "val", "test"):
            images = iter_split_images(args.data, split)
            print(f"{split}: {len(images)} images")
            if len(images) != EXPECTED_COUNTS[split]:
                msg = f"{split} count expected {EXPECTED_COUNTS[split]}, got {len(images)}"
                if args.strict:
                    raise SystemExit(msg)
                print("WARNING:", msg)
            for image in images:
                label = label_path_for_image(image)
                rows = read_yolo_labels(label)
                total_boxes += len(rows)
                bad = [r for r in rows if r[0] < 0 or r[0] >= max(1, len(names))]
                if bad:
                    raise ValueError(f"Out-of-range class id in {label}: {bad[:3]}")
                try:
                    image_id = image.relative_to(dataset_root).as_posix()
                except ValueError:
                    image_id = image.name
                try:
                    label_id = label.relative_to(dataset_root).as_posix()
                except ValueError:
                    label_id = label.name
                w.writerow([
                    split,
                    image_id,
                    label_id,
                    len(rows),
                    sha256(image) if args.hash_images else "",
                ])
    print(f"manifest: {manifest} | total boxes: {total_boxes}")


if __name__ == "__main__":
    main()
