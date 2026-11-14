"""Crop ground-truth text boxes for TrOCR training (paper section III-D).

Only FC_A carries transcriptions (FC_B text labels are empty), as in the paper.
A fixed share of FC_A train images (by writer) is held out for validation.

Output:
    data_work/text/fca/<split>/<name>.png
    data_work/text/fca/<split>.tsv          (file<TAB>text, paths relative to the tsv)
Usage: python -m hdr.data.text_crops
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from hdr import config
from hdr.data.coco import load_split


def crop(gray, bbox, pad: int = 4):
    x, y, w, h = (int(round(v)) for v in bbox)
    H, W = gray.shape[:2]
    return gray[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]


def writer_of(file_name: str) -> str:
    return file_name.split("_")[0]


def export(images, out_dir: Path, tsv: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for im in images:
        gray = cv2.imread(str(im.path), cv2.IMREAD_GRAYSCALE)
        for a in im.annotations:
            if a.category != "text" or not a.text or not a.text.strip():
                continue
            c = crop(gray, a.bbox)
            if c.size == 0:
                continue
            name = f"{Path(im.file_name).stem}_{a.id}.png"
            cv2.imwrite(str(out_dir / name), c)
            rows.append(f"{out_dir.name}/{name}\t{a.text.strip()}")
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.TEXT_ROOT / "fca")
    ap.add_argument("--val-writers", type=float, default=0.1,
                    help="share of FC_A train writers held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = load_split("fca", "train")
    writers = sorted({writer_of(im.file_name) for im in train})
    random.Random(args.seed).shuffle(writers)
    val_w = set(writers[: max(1, int(len(writers) * args.val_writers))])
    splits = {
        "train": [im for im in train if writer_of(im.file_name) not in val_w],
        "val": [im for im in train if writer_of(im.file_name) in val_w],
        "test": load_split("fca", "test"),
    }
    for name, ims in splits.items():
        n = export(ims, args.out / name, args.out / f"{name}.tsv")
        print(f"fca/{name}: {n} crops from {len(ims)} images")


if __name__ == "__main__":
    main()
