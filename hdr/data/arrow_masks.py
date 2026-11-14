"""Draw binary arrow masks from annotated waypoints (paper section III-C).

Each arrow polyline is drawn with a fixed stroke width (6 px in the paper).
Mask pixels are 255 for arrow and 0 for everything else.

Only FC_A and FC_B carry waypoints. The paper trains one DeepLabv3 on both
and reuses it for the FA dataset.

Output: data_work/arrow_masks/<dataset>/<split>/<stem>.png
Usage: python -m hdr.data.arrow_masks --datasets fca fcb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hdr import config
from hdr.data.coco import ImageRecord, available_splits, load_split


NODE_CATEGORIES = {"connection", "data", "decision", "process", "terminator", "state", "final state"}


def polyline_mask(im: ImageRecord, width: int = config.ARROW_MASK_WIDTH) -> np.ndarray:
    """Paper method: waypoints drawn with a fixed stroke width."""
    mask = np.zeros((im.height, im.width), dtype=np.uint8)
    for a in im.annotations:
        if a.category != "arrow" or not a.waypoints:
            continue
        pts = np.round(np.asarray(a.waypoints, dtype=np.float64)).astype(np.int32)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], isClosed=False, color=255,
                      thickness=width, lineType=cv2.LINE_8)
    return mask


def ink(gray: np.ndarray) -> np.ndarray:
    """Binary ink map (255 = dark stroke) via Otsu thresholding."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _parallelogram(inkmap: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Estimate a data symbol's slanted outline from ink in its top and bottom rows."""
    bh = y1 - y0
    band = max(2, bh // 7)

    def left_edge(ys: slice) -> float | None:
        xs = np.nonzero(inkmap[ys, x0:x1])[1]
        return float(np.percentile(xs, 2)) + x0 if len(xs) else None

    top, bot = left_edge(slice(y0, y0 + band)), left_edge(slice(y1 - band, y1))
    if top is None or bot is None or abs(top - bot) < 0.05 * (x1 - x0):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
    s = int(abs(top - bot))
    if top > bot:   # leans right: /‾‾/
        return np.array([[x0 + s, y0], [x1, y0], [x1 - s, y1], [x0, y1]], np.int32)
    return np.array([[x0, y0], [x1 - s, y0], [x1, y1], [x0 + s, y1]], np.int32)


def node_region(a, inkmap: np.ndarray, grow: int = 3) -> np.ndarray:
    """Mask of a symbol's real shape (outline included), not its bounding box.

    Removing the whole box would also erase arrowheads that end in the empty
    corners of diamonds, ellipses and slanted parallelograms.
    """
    h, w = inkmap.shape
    x0, y0, bw, bh = (int(round(v)) for v in a.bbox)
    x1, y1 = min(w - 1, x0 + bw), min(h - 1, y0 + bh)
    x0, y0 = max(0, x0), max(0, y0)
    m = np.zeros((h, w), np.uint8)
    if a.category == "decision":
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.fillPoly(m, [np.array([[cx, y0], [x1, cy], [cx, y1], [x0, cy]], np.int32)], 255)
    elif a.category == "terminator":
        # Drawn as a capsule (stadium) or an ellipse; the capsule covers both.
        r = max(1, min(x1 - x0, y1 - y0) // 2)
        m[y0:y1 + 1, x0 + r:x1 - r + 1] = 255
        m[y0 + r:y1 - r + 1, x0:x1 + 1] = 255
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(m, (cx, cy), r, 255, -1)
    elif a.category in ("connection", "state", "final state"):
        # RotatedRect form: ((cx, cy), (full width, full height), angle)
        cv2.ellipse(m, (((x0 + x1) / 2, (y0 + y1) / 2), (x1 - x0, y1 - y0), 0), 255, -1)
    elif a.category == "data":
        cv2.fillPoly(m, [_parallelogram(inkmap, x0, y0, x1, y1)], 255)
    else:  # process: the box fits the outline exactly, no growth needed
        m[y0:y1 + 1, x0:x1 + 1] = 255
        return m
    if grow:
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return m


def ink_mask(im: ImageRecord, gray: np.ndarray, band: int = 15,
             head_frac: float = 0.035) -> np.ndarray:
    """Ink pixels that belong to arrows.

    The offline waypoints (especially FC_B) are a coarse line through the
    shaft: they miss the arrowhead barbs and drift off the ink at corners.
    So: take a wide band around the waypoints plus a disc around the head
    keypoint, keep only real ink inside it, and drop ink that belongs to the
    symbols the arrow touches (their real shape, see node_region).
    """
    inkmap = ink(gray)
    h, w = gray.shape
    region = np.zeros((h, w), dtype=np.uint8)
    r = max(8, int(head_frac * max(h, w)))
    hi = config.ARROW_HEAD_KP_INDEX
    for a in im.annotations:
        if a.category != "arrow" or not a.waypoints:
            continue
        pts = np.round(np.asarray(a.waypoints, dtype=np.float64)).astype(np.int32)
        cv2.polylines(region, [pts.reshape(-1, 1, 2)], False, 255, thickness=band)
        if a.keypoints:
            hx, hy = a.keypoints[3 * hi], a.keypoints[3 * hi + 1]
            cv2.circle(region, (int(round(hx)), int(round(hy))), r, 255, -1)
    for a in im.annotations:
        if a.category in NODE_CATEGORIES:
            region[node_region(a, inkmap) > 0] = 0
    mask = cv2.bitwise_and(inkmap, region)
    if not mask.any():
        mask = polyline_mask(im)
    return mask


def arrow_mask(im: ImageRecord, mode: str = "ink") -> np.ndarray:
    if mode == "polyline":
        return polyline_mask(im)
    gray = cv2.imread(str(im.path), cv2.IMREAD_GRAYSCALE)
    return ink_mask(im, gray)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["fca", "fcb"])
    ap.add_argument("--out", type=Path, default=config.MASK_ROOT)
    ap.add_argument("--mode", choices=["ink", "polyline"], default="ink",
                    help="ink: waypoint band AND real ink (default); polyline: paper's 6px line")
    args = ap.parse_args()
    for ds in args.datasets:
        for split in available_splits(ds):
            out = args.out / ds / split
            out.mkdir(parents=True, exist_ok=True)
            images = load_split(ds, split)
            missing = 0
            for im in images:
                arrows = [a for a in im.annotations if a.category == "arrow"]
                missing += sum(1 for a in arrows if not a.waypoints)
                cv2.imwrite(str(out / f"{Path(im.file_name).stem}.png"), arrow_mask(im, args.mode))
            print(f"{ds}/{split}: {len(images)} masks, arrows without waypoints: {missing}")


if __name__ == "__main__":
    main()
