"""Draw ground-truth annotations on images for a visual sanity check.

Boxes are coloured per class. Arrow keypoint 0 is a green dot, keypoint 1 a red
dot, waypoints a blue polyline.

Usage: python -m hdr.data.viz_annotations --dataset fca --split train --n 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from hdr.config import DATASET_CLASSES, OUT_ROOT, YOLO_DATA_ROOT
from hdr.data.coco import load_split

COLORS = {
    "arrow": (0, 140, 255), "connection": (128, 0, 128), "data": (255, 0, 0),
    "decision": (0, 128, 0), "process": (200, 120, 0), "terminator": (0, 0, 200),
    "text": (120, 120, 120), "state": (255, 0, 0), "final state": (0, 128, 0),
    "arrow_head": (0, 0, 255), "arrow_tail": (0, 200, 0),
}


def draw(im) -> "cv2.Mat":
    img = cv2.imread(str(im.path))
    for a in im.annotations:
        x, y, w, h = map(int, a.bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), COLORS.get(a.category, (0, 0, 0)), 2)
        if a.category == "arrow":
            if a.waypoints:
                pts = [tuple(map(int, p)) for p in a.waypoints]
                for p, q in zip(pts, pts[1:]):
                    cv2.line(img, p, q, (255, 0, 0), 1)
            if a.keypoints:
                k0 = tuple(map(int, a.keypoints[0:2]))
                k1 = tuple(map(int, a.keypoints[3:5]))
                cv2.circle(img, k0, 7, (0, 200, 0), -1)
                cv2.circle(img, k1, 7, (0, 0, 255), -1)
    return img


def draw_yolo(img_path: Path, label_path: Path, classes: list[str]):
    """Draw YOLO txt labels back onto the image (checks the conversion)."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        name = classes[int(c)]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p0 = (int(cx - bw / 2), int(cy - bh / 2))
        p1 = (int(cx + bw / 2), int(cy + bh / 2))
        cv2.rectangle(img, p0, p1, COLORS.get(name, (0, 0, 0)), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fca")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--yolo", action="store_true", help="draw converted YOLO labels")
    args = ap.parse_args()
    out = OUT_ROOT / ("viz_yolo" if args.yolo else "viz") / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    for im in load_split(args.dataset, args.split)[: args.n]:
        if args.yolo:
            base = YOLO_DATA_ROOT / args.dataset
            lbl = base / "labels" / args.split / f"{Path(im.file_name).stem}.txt"
            img = draw_yolo(im.path, lbl, DATASET_CLASSES[args.dataset])
        else:
            img = draw(im)
        cv2.imwrite(str(out / im.file_name), img)
    print("wrote", out)


if __name__ == "__main__":
    main()
