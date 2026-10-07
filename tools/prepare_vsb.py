"""Convert the public VSB wood-defect bounding boxes to YOLO format.

VSB bounding-box files use one record per object, e.g.:
    Knot_OK 0,421786 0,819336 0,571429 1,000000
The four normalized coordinates are interpreted as left, top, right, bottom.

This script creates a deterministic 2,800-image seven-class subset (seed=42 by
default), freezes the exact selected filenames to CSV, and writes a
YOLO data.yaml. Unknown/excluded labels are always reported rather than silently
mapped.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml
import cv2

from agm_yolo.data import IMAGE_EXTENSIONS

TARGET_CLASSES = [
    "Live Knot",
    "Dead Knot",
    "Marrow",
    "Resin Pocket",
    "Knot with Crack",
    "Knot Missing",
    "Crack",
]

# Normalization intentionally accepts the spellings used in public VSB copies
# plus human-readable variants. Quartzity, Blue stain and Overgrown are excluded
# by the seven-class AGM-YOLO protocol.
ALIASES = {
    "knotok": 0,
    "liveknot": 0,
    "knotnok": 1,
    "deadknot": 1,
    "marrow": 2,
    "resin": 3,
    "resinpocket": 3,
    "knotcrack": 4,
    "knotwithcrack": 4,
    "knotmissing": 5,
    "missingknot": 5,
    "crack": 6,
}
EXCLUDED = {"quartzity", "bluestain", "overgrown"}


def token(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_annotation(path: Path) -> tuple[list[tuple[int, float, float, float, float]], Counter, Counter]:
    labels: list[tuple[int, float, float, float, float]] = []
    excluded = Counter()
    unknown = Counter()
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected 5 fields at {path}:{line_no}, got: {line!r}")
        raw_class, *coords = fields
        key = token(raw_class)
        if key in EXCLUDED:
            excluded[raw_class] += 1
            continue
        if key not in ALIASES:
            unknown[raw_class] += 1
            continue
        left, top, right, bottom = map(parse_num, coords)
        # Public description says coordinates are percentage/100 and the example
        # is consistent with L,T,R,B. Clamp tiny rounding excursions only.
        vals = [left, top, right, bottom]
        if any(v < -1e-6 or v > 1.000001 for v in vals):
            raise ValueError(f"Coordinate outside [0,1] at {path}:{line_no}: {vals}")
        left, top, right, bottom = [min(1.0, max(0.0, v)) for v in vals]
        if right <= left or bottom <= top:
            raise ValueError(
                f"Invalid LTRB box at {path}:{line_no}: {(left, top, right, bottom)}; "
                "check raw annotation order before proceeding"
            )
        xc = (left + right) / 2.0
        yc = (top + bottom) / 2.0
        w = right - left
        h = bottom - top
        labels.append((ALIASES[key], xc, yc, w, h))
    return labels, excluded, unknown


def build_annotation_index(label_root: Path) -> dict[str, Path]:
    """Index annotations once; avoids an O(images × files) recursive search."""
    index: dict[str, Path] = {}
    for p in sorted(label_root.rglob("*.txt")):
        stem = p.stem[:-5] if p.stem.endswith("_anno") else p.stem
        if stem in index:
            raise RuntimeError(f"Ambiguous annotation stem {stem!r}: {index[stem]} and {p}")
        index[stem] = p
    return index


def find_annotation(annotation_index: dict[str, Path], image: Path) -> Path | None:
    return annotation_index.get(image.stem)


def link_or_copy(src: Path, dst: Path, mode: str, resize: tuple[int, int] | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if resize is not None:
        width, height = resize
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(src)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(dst), image):
            raise RuntimeError(f"Failed to write resized image: {dst}")
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--images", required=True, help="Root containing VSB Images*/ image folders")
    p.add_argument("--boxes", required=True, help="Root containing Bounding_Boxes annotations")
    p.add_argument("--out", default="datasets/vsb_7class_seed42")
    p.add_argument("--subset-size", type=int, default=2800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.30, help="Manuscript 7:3 train-pool:test")
    p.add_argument("--val-fraction-of-train", type=float, default=0.10,
                   help="Validation held from the 70%% train pool; set 0 to use no val split")
    p.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    p.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None,
                   help="Materialize each selected image at an exact size; use 1280 512 for the AGM-YOLO VSB protocol")
    p.add_argument("--drop-empty", action="store_true",
                   help="Exclude images with no retained seven-class boxes before sampling")
    args = p.parse_args()

    image_root, box_root, out = Path(args.images), Path(args.boxes), Path(args.out)
    images = sorted(
        (p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.relative_to(image_root).as_posix(),
    )
    if not images:
        raise SystemExit(f"No images found under {image_root}")

    annotation_index = build_annotation_index(box_root)
    records = []
    excluded_total, unknown_total = Counter(), Counter()
    missing_boxes = []
    for img in images:
        ann = find_annotation(annotation_index, img)
        if ann is None:
            missing_boxes.append(img.relative_to(image_root).as_posix())
            continue
        labels, excluded, unknown = parse_annotation(ann)
        excluded_total.update(excluded)
        unknown_total.update(unknown)
        if labels or not args.drop_empty:
            records.append((img, ann, labels))

    if unknown_total:
        raise SystemExit(
            "Unknown VSB labels were found. Add an explicit mapping before conversion: "
            + json.dumps(dict(unknown_total), ensure_ascii=False)
        )
    if len(records) < args.subset_size:
        raise SystemExit(f"Only {len(records)} eligible images, need {args.subset_size}")

    rng = random.Random(args.seed)
    chosen = rng.sample(records, args.subset_size)
    # Freeze split after selection with a second deterministic shuffle.
    rng.shuffle(chosen)
    n_test = round(args.subset_size * args.test_fraction)
    n_train_pool = args.subset_size - n_test
    train_pool = chosen[:n_train_pool]
    test = chosen[n_train_pool:]
    n_val = round(n_train_pool * args.val_fraction_of_train)
    val = train_pool[-n_val:] if n_val else []
    train = train_pool[:-n_val] if n_val else train_pool
    splits = {"train": train, "val": val, "test": test}

    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    counts = Counter()
    for split, subset in splits.items():
        if not subset and split == "val":
            continue
        for index, (img, ann, labels) in enumerate(subset):
            # Prefix the deterministic index to make basename collisions impossible.
            dst_name = f"{index:05d}_{img.stem}{img.suffix.lower()}"
            dst_img = out / "images" / split / dst_name
            dst_lab = out / "labels" / split / (Path(dst_name).stem + ".txt")
            link_or_copy(img, dst_img, args.mode, tuple(args.resize) if args.resize else None)
            dst_lab.parent.mkdir(parents=True, exist_ok=True)
            dst_lab.write_text(
                "".join(f"{c} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}\n" for c, xc, yc, w, h in labels),
                encoding="utf-8",
            )
            for c, *_ in labels:
                counts[TARGET_CLASSES[c]] += 1
            manifest_rows.append({
                "split": split,
                "source_image": img.relative_to(image_root).as_posix(),
                "source_annotation": ann.relative_to(box_root).as_posix(),
                "output_image": dst_img.relative_to(out).as_posix(),
                "retained_boxes": len(labels),
            })

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val" if val else "images/train",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(TARGET_CLASSES)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        w.writeheader(); w.writerows(manifest_rows)

    report = {
        "seed": args.seed,
        "subset_size": args.subset_size,
        "train_pool_before_val": n_train_pool,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "class_box_counts": dict(counts),
        "excluded_raw_labels": dict(excluded_total),
        "missing_annotation_files": len(missing_boxes),
        "source_images_seen": len(images),
        "eligible_images": len(records),
        "drop_empty_before_sampling": args.drop_empty,
        "materialized_resize": args.resize,
        "note": "Deterministic split generated from sorted source paths and the selected random seed.",
    }
    (out / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"data.yaml: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
