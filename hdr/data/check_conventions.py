"""Measure what arrow keypoints and arrow_prev / arrow_next refer to.

For every arrow, look up the prev/next symbol by annotation `id` and by `aid`,
then measure how far each keypoint lies from each symbol box. The keypoint that
lies at the `arrow_next` symbol is the head.

Usage: python -m hdr.data.check_conventions
"""
from __future__ import annotations

from collections import Counter

from hdr.data.coco import available_splits, load_split


def box_dist(pt, bbox) -> float:
    """Distance from a point to an axis-aligned box (0 if inside)."""
    x, y = pt
    bx, by, bw, bh = bbox
    dx = max(bx - x, 0, x - (bx + bw))
    dy = max(by - y, 0, y - (by + bh))
    return (dx * dx + dy * dy) ** 0.5


def check(dataset: str) -> dict:
    stats = Counter()
    for split in available_splits(dataset):
        for im in load_split(dataset, split):
            by_id = {a.id: a for a in im.annotations}
            by_aid = {a.aid: a for a in im.annotations if a.aid is not None}
            for a in im.annotations:
                if a.category != "arrow" or not a.keypoints:
                    continue
                stats["arrows"] += 1
                kp0 = a.keypoints[0:2]
                kp1 = a.keypoints[3:5]
                if a.waypoints:
                    stats["wp_starts_at_kp0"] += a.waypoints[0] == list(kp0)
                for key, lut in (("id", by_id), ("aid", by_aid)):
                    nxt = lut.get(a.arrow_next)
                    prv = lut.get(a.arrow_prev)
                    if nxt is None or nxt.category == "arrow":
                        continue
                    stats[f"{key}_next_found"] += 1
                    d0, d1 = box_dist(kp0, nxt.bbox), box_dist(kp1, nxt.bbox)
                    stats[f"{key}_next_nearer_kp0"] += d0 < d1
                    stats[f"{key}_next_nearer_kp1"] += d1 < d0
                    if prv is not None and prv.category != "arrow":
                        stats[f"{key}_prev_found"] += 1
                        stats[f"{key}_prev_nearer_kp0"] += box_dist(kp0, prv.bbox) < box_dist(kp1, prv.bbox)
    return dict(stats)


def main() -> None:
    for ds in ("fca", "fcb", "fa"):
        print(ds, check(ds))


if __name__ == "__main__":
    main()
