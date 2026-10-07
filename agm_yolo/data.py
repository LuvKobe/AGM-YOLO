"""Dataset utilities shared by AGM-YOLO reproduction scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def load_data_yaml(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["_yaml_path"] = str(path)
    return data


def class_names(data: dict) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(x) for x in names]


def resolve_dataset_root(data: dict) -> Path:
    yaml_path = Path(data["_yaml_path"])
    root = data.get("path")
    if root is None:
        return yaml_path.parent
    root = Path(str(root)).expanduser()
    return root.resolve() if root.is_absolute() else (yaml_path.parent / root).resolve()


def resolve_split(data: dict, split: str) -> list[Path]:
    """Resolve a YOLO data.yaml split into one or more paths.

    Supports directories, image-list text files, and YAML lists. Relative split
    paths are resolved against ``path:`` when present, otherwise the YAML file.
    """
    value = data.get(split)
    if value is None:
        raise KeyError(f"Split {split!r} is missing from data YAML")
    values = value if isinstance(value, list) else [value]
    root = resolve_dataset_root(data)
    out: list[Path] = []
    for item in values:
        p = Path(str(item)).expanduser()
        if not p.is_absolute():
            p = root / p
        out.append(p.resolve())
    return out


def iter_images_from_path(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p.resolve()
        return
    if path.is_file() and path.suffix.lower() == ".txt":
        base = path.parent
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            p = Path(s).expanduser()
            if not p.is_absolute():
                p = (base / p).resolve()
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p
        return
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        yield path.resolve()
        return
    raise FileNotFoundError(f"Unsupported or missing split path: {path}")


def iter_split_images(data_yaml: str | Path, split: str = "test") -> list[Path]:
    data = load_data_yaml(data_yaml)
    images: list[Path] = []
    for split_path in resolve_split(data, split):
        images.extend(iter_images_from_path(split_path))
    # Deterministic ordering, de-duplicated without changing first occurrence.
    return list(dict.fromkeys(images))


def label_path_for_image(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    parts = list(image_path.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image_path.with_suffix(".txt")


def read_yolo_labels(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Malformed YOLO label at {label_path}:{line_no}: {line!r}")
        cls, xc, yc, w, h = fields[:5]
        rows.append((int(float(cls)), float(xc), float(yc), float(w), float(h)))
    return rows
