"""Convert COCO diagram annotations to YOLO txt format.

Adds two classes per arrow, `arrow_head` and `arrow_tail`: square boxes centred
on the arrow keypoints (paper section III-B).

Output layout (what YOLOv7 expects):
    data_yolo/<dataset>/images/<split>/*.png
    data_yolo/<dataset>/labels/<split>/*.txt
    data_yolo/<dataset>.yaml

FC_A has no validation split, so its yaml points `val` at FC_B val, as in the
paper (section IV-A).

Usage: python -m hdr.data.coco_to_yolo --datasets fca fcb fa
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from hdr import config
from hdr.data.coco import ImageRecord, available_splits, load_split

Box = tuple[float, float, float, float]  # x, y, w, h in pixels


def endpoint_box(x: float, y: float, img_w: int, img_h: int,
                 frac: float = config.ENDPOINT_BOX_FRAC,
                 min_px: int = config.ENDPOINT_BOX_MIN_PX) -> Box:
    """Square box centred on (x, y), clipped to the image."""
    side = max(min_px, frac * max(img_w, img_h))
    return clip_box((x - side / 2, y - side / 2, side, side), img_w, img_h)


def clip_box(box: Box, img_w: int, img_h: int) -> Box:
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def to_yolo_line(cls_id: int, box: Box, img_w: int, img_h: int) -> str:
    x, y, w, h = box
    cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def image_labels(im: ImageRecord, classes: list[str]) -> list[str]:
    cls_ids = {name: i for i, name in enumerate(classes)}
    lines = []
    for a in im.annotations:
        if a.category not in cls_ids:
            raise KeyError(f"{im.file_name}: unknown category {a.category!r}")
        box = clip_box(a.bbox, im.width, im.height)
        if box[2] <= 0 or box[3] <= 0:
            continue
        lines.append(to_yolo_line(cls_ids[a.category], box, im.width, im.height))
        if a.category == "arrow" and a.keypoints:
            for name, idx in (("arrow_head", config.ARROW_HEAD_KP_INDEX),
                              ("arrow_tail", config.ARROW_TAIL_KP_INDEX)):
                kx, ky = a.keypoints[3 * idx], a.keypoints[3 * idx + 1]
                ebox = endpoint_box(kx, ky, im.width, im.height)
                lines.append(to_yolo_line(cls_ids[name], ebox, im.width, im.height))
    return lines


def convert_split(dataset: str, split: str, out_root: Path) -> int:
    classes = config.DATASET_CLASSES[dataset]
    img_dir = out_root / dataset / "images" / split
    lbl_dir = out_root / dataset / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    images = load_split(dataset, split)
    for im in images:
        dst = img_dir / im.file_name
        if not dst.exists():
            shutil.copy2(im.path, dst)
        (lbl_dir / f"{Path(im.file_name).stem}.txt").write_text(
            "\n".join(image_labels(im, classes)) + "\n", encoding="utf-8")
    return len(images)


def write_yaml(dataset: str, out_root: Path) -> Path:
    splits = available_splits(dataset)
    img = lambda ds, sp: str((out_root / ds / "images" / sp).resolve()).replace("\\", "/")
    cfg = {
        "train": img(dataset, "train"),
        "val": img(dataset, "val") if "val" in splits else img("fcb", "val"),
        "test": img(dataset, "test"),
        "nc": len(config.DATASET_CLASSES[dataset]),
        "names": config.DATASET_CLASSES[dataset],
    }
    path = out_root / f"{dataset}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["fca", "fcb", "fa"])
    ap.add_argument("--out", type=Path, default=config.YOLO_DATA_ROOT)
    args = ap.parse_args()
    todo = set(args.datasets)
    if "fca" in todo:
        todo.add("fcb")  # FC_A borrows FC_B val
    for ds in sorted(todo):
        for split in available_splits(ds):
            n = convert_split(ds, split, args.out)
            print(f"{ds}/{split}: {n} images")
        print("yaml:", write_yaml(ds, args.out))


if __name__ == "__main__":
    main()
