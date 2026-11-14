"""Post-processing of YOLOv7 detections (paper section III-C).

Steps, in the paper's order:
 1. Split arrows from the other classes.
 2. Non-arrow symbols: same-class NMS, overlap > GENERAL_THRESHOLD (0.5) is a duplicate.
 3. Arrows: NMS at ARROW_THRESHOLD (0.6). Arrow pairs that overlap less than that
    are checked on the DeepLab mask with connected components: boxes covering the
    same arrow component are duplicates (keep the one covering more of it), boxes
    covering different components are separate arrows.
 4. Give each arrow a head and a tail. Use detected arrow_head / arrow_tail boxes;
    if one is missing, fall back to skeleton endpoints (Otsu + Zhang-Suen) of the
    arrow's ink inside its box.
 5. Attach text to symbols: terminator, data, process and decision each get the
    text closest to their centre (here: text inside the box, else the nearest one).
    Unattached text is kept as possible arrow labels (yes/no, transitions).

All functions are pure (numpy in, dataclasses out) so they can be unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from hdr import config
from hdr.infer.models import Detection

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

NODE_CLASSES = {"connection", "data", "decision", "process", "terminator", "state", "final state"}
TEXT_NODES = {"terminator", "data", "process", "decision"}   # must carry text (paper)
ARROW, HEAD, TAIL, TEXT = "arrow", "arrow_head", "arrow_tail", "text"


# ---------------------------------------------------------------- geometry

def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    i = intersection(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def box_distance(a: Box, b: Box) -> float:
    """Gap between two boxes (0 if they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def contains_point(b: Box, p: tuple[float, float], margin: float = 0.0) -> bool:
    return b[0] - margin <= p[0] <= b[2] + margin and b[1] - margin <= p[1] <= b[3] + margin


def square(p: tuple[float, float], side: float) -> Box:
    return (p[0] - side / 2, p[1] - side / 2, p[0] + side / 2, p[1] + side / 2)


# ---------------------------------------------------------------- results

@dataclass
class Arrow:
    box: Box
    conf: float
    head: Box | None = None          # small box at the pointing end
    tail: Box | None = None          # small box at the start
    head_source: str = ""            # "detected" or "skeleton"
    tail_source: str = ""
    head_conf: float = 0.0           # detector score of the head box (low for skeleton)
    tail_conf: float = 0.0


@dataclass
class Symbol:
    cls: str
    box: Box
    conf: float
    text_boxes: list[Box] = field(default_factory=list)
    text: str = ""


@dataclass
class Cleaned:
    symbols: list[Symbol]
    arrows: list[Arrow]
    free_texts: list[Box]            # text not inside a symbol: arrow labels
    removed: dict[str, int]          # how many duplicates each rule removed


# ---------------------------------------------------------------- steps

def nms(dets: list[Detection], thr: float) -> list[Detection]:
    """Same-class NMS: drop the lower-confidence box when overlap > thr."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(k.cls != d.cls or iou(k.box, d.box) <= thr for k in kept):
            kept.append(d)
    return kept


def _dominant_component(labels: np.ndarray, box: Box) -> tuple[int, int]:
    """(component label, pixel count) that covers most of the mask inside box."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    sub = labels[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    vals, counts = np.unique(sub[sub > 0], return_counts=True)
    if not len(vals):
        return 0, 0
    i = int(np.argmax(counts))
    return int(vals[i]), int(counts[i])


def stroke_labels(mask: np.ndarray | None) -> np.ndarray | None:
    """Connected components of the arrow mask (one label per arrow stroke)."""
    if mask is None or not mask.any():
        return None
    # small closing so a stroke broken by 1-2 px still counts as one arrow
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.connectedComponents(m)[1]


def dedupe_arrows(arrows: list[Detection], mask: np.ndarray | None,
                  thr: float = config.ARROW_THRESHOLD, min_overlap: float = 0.3) -> list[Detection]:
    """NMS at thr, then boxes overlapping by at least `min_overlap` that cover the
    same stroke are duplicates. Below min_overlap the boxes belong to different
    arrows even if a big box (a long arc) happens to contain another arrow's stroke."""
    arrows = nms(arrows, thr)
    labels = stroke_labels(mask)
    if labels is None or len(arrows) < 2:
        return arrows
    dom = [_dominant_component(labels, a.box) for a in arrows]
    drop: set[int] = set()
    for i in range(len(arrows)):
        for j in range(i + 1, len(arrows)):
            if i in drop or j in drop or iou(arrows[i].box, arrows[j].box) < min_overlap:
                continue
            (li, ci), (lj, cj) = dom[i], dom[j]
            if li and li == lj:  # both boxes cover the same arrow stroke
                drop.add(j if ci >= cj else i)
    return [a for k, a in enumerate(arrows) if k not in drop]


def _skeleton_endpoints(ink: np.ndarray) -> list[tuple[int, int]]:
    """Endpoints (x, y) of the 1-px skeleton: pixels with exactly one neighbour."""
    from skimage.morphology import skeletonize
    sk = skeletonize(ink > 0).astype(np.uint8)
    if not sk.any():
        return []
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    ys, xs = np.nonzero((sk == 1) & (nb == 1))
    return list(zip(xs.tolist(), ys.tolist()))


def skeleton_ends(gray: np.ndarray, mask: np.ndarray | None, box: Box
                  ) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Two farthest skeleton endpoints of the arrow's ink inside box (paper fallback)."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    _, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask is not None:
        ink = cv2.bitwise_and(ink, (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255)
    ends = _skeleton_endpoints(ink)
    if len(ends) < 2:
        return None
    pts = np.array(ends, dtype=np.float64)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return (pts[i][0] + x1, pts[i][1] + y1), (pts[j][0] + x1, pts[j][1] + y1)


def _ink_near(gray: np.ndarray, p: tuple[float, float], r: int) -> int:
    x, y = int(p[0]), int(p[1])
    crop = gray[max(0, y - r):y + r, max(0, x - r):x + r]
    return int((crop < 128).sum()) if crop.size else 0


def _labels_near(labels: np.ndarray, p: tuple[float, float], r: int) -> set[int]:
    x, y = int(p[0]), int(p[1])
    sub = labels[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return set(np.unique(sub[sub > 0]).tolist())


def assign_ends(arrows: list[Detection], heads: list[Detection], tails: list[Detection],
                labels: np.ndarray | None, side: float
                ) -> tuple[dict[int, Detection], dict[int, Detection]]:
    """Choose each arrow's head and tail together.

    Per arrow, every (head, tail) combination inside its box is scored by:
    the two detector confidences, +1 for each end lying on the arrow's own
    stroke (same mask component), and how far apart the two ends are relative
    to the box size (an arrow's ends sit at opposite ends of its box; this is
    what separates two arrows drawn in opposite directions between the same
    pair of symbols). Combinations are then taken best-first, each detected
    end used by one arrow only. Without this, overlapping arrows borrow each
    other's ends. A second pass lets an arrow that got nothing reuse ends on
    its own stroke, because two arrows can genuinely meet at one point.
    A missing head or tail is allowed (it is found later by the skeleton).
    """
    margin = side / 2
    dom = [_dominant_component(labels, a.box)[0] if labels is not None else 0 for a in arrows]
    on_stroke = lambda i, c: bool(dom[i]) and dom[i] in _labels_near(labels, c.center, int(side / 2))
    triples = []
    for i, a in enumerate(arrows):
        diag = max(1.0, float(np.hypot(a.box[2] - a.box[0], a.box[3] - a.box[1])))
        hs = [j for j, h in enumerate(heads) if contains_point(a.box, h.center, margin)] + [None]
        ts = [j for j, t in enumerate(tails) if contains_point(a.box, t.center, margin)] + [None]
        for hj in hs:
            for tj in ts:
                if hj is None and tj is None:
                    continue
                s = 0.0
                for j, cands in ((hj, heads), (tj, tails)):
                    if j is not None:
                        s += cands[j].conf + (1.0 if on_stroke(i, cands[j]) else 0.0)
                if hj is not None and tj is not None:
                    (hx, hy), (tx, ty) = heads[hj].center, tails[tj].center
                    s += min(1.0, float(np.hypot(hx - tx, hy - ty)) / diag)
                triples.append((s, i, hj, tj))
    head_of: dict[int, Detection] = {}
    tail_of: dict[int, Detection] = {}
    done: set[int] = set()
    used_h: set[int] = set()
    used_t: set[int] = set()
    for _, i, hj, tj in sorted(triples, key=lambda x: -x[0]):
        if i in done or (hj is not None and hj in used_h) or (tj is not None and tj in used_t):
            continue
        done.add(i)
        if hj is not None:
            head_of[i] = heads[hj]
            used_h.add(hj)
        if tj is not None:
            tail_of[i] = tails[tj]
            used_t.add(tj)
    for _, i, hj, tj in sorted(triples, key=lambda x: -x[0]):   # reuse, own stroke only
        if i in done:
            continue
        if all(j is None or on_stroke(i, c[j]) for j, c in ((hj, heads), (tj, tails))):
            done.add(i)
            if hj is not None:
                head_of[i] = heads[hj]
            if tj is not None:
                tail_of[i] = tails[tj]
    return head_of, tail_of


def attach_endpoints(arrows: list[Detection], heads: list[Detection], tails: list[Detection],
                     gray: np.ndarray, mask: np.ndarray | None, side: float) -> list[Arrow]:
    out = []
    labels = stroke_labels(mask)
    head_of, tail_of = assign_ends(arrows, heads, tails, labels, side)
    for k, a in enumerate(arrows):
        arrow = Arrow(a.box, a.conf)
        if k in head_of:
            h = head_of[k]
            arrow.head, arrow.head_source, arrow.head_conf = h.box, "detected", h.conf
        if k in tail_of:
            t = tail_of[k]
            arrow.tail, arrow.tail_source, arrow.tail_conf = t.box, "detected", t.conf
        if arrow.head and arrow.tail and iou(arrow.head, arrow.tail) > 0.3:
            # same spot: keep the more confident one, find the other end again
            if arrow.tail_conf > arrow.head_conf:
                arrow.head, arrow.head_source, arrow.head_conf = None, "", 0.0
            else:
                arrow.tail, arrow.tail_source, arrow.tail_conf = None, "", 0.0
        if arrow.head is None or arrow.tail is None:
            ends = skeleton_ends(gray, mask, a.box)
            if ends is None:  # last resort: ends of the box's long axis
                x1, y1, x2, y2 = a.box
                ends = (((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2)) if (y2 - y1) >= (x2 - x1) \
                    else ((x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2))
            e0, e1 = ends
            dist = lambda p, b: np.hypot(p[0] - center(b)[0], p[1] - center(b)[1])
            fallback_conf = 0.1 * a.conf        # rank skeleton guesses below detections
            if arrow.head is not None:          # tail = the end farther from the head
                tail_p = e0 if dist(e0, arrow.head) > dist(e1, arrow.head) else e1
                arrow.tail, arrow.tail_source, arrow.tail_conf = square(tail_p, side), "skeleton", fallback_conf
            elif arrow.tail is not None:        # head = the end farther from the tail
                head_p = e0 if dist(e0, arrow.tail) > dist(e1, arrow.tail) else e1
                arrow.head, arrow.head_source, arrow.head_conf = square(head_p, side), "skeleton", fallback_conf
            else:                               # arrowhead end has more ink (the barbs)
                r = int(side)
                head_p, tail_p = (e0, e1) if _ink_near(gray, e0, r) >= _ink_near(gray, e1, r) else (e1, e0)
                arrow.head, arrow.head_source, arrow.head_conf = square(head_p, side), "skeleton", fallback_conf
                arrow.tail, arrow.tail_source, arrow.tail_conf = square(tail_p, side), "skeleton", fallback_conf
        out.append(arrow)
    return out


def attach_text(symbols: list[Symbol], texts: list[Detection]) -> list[Box]:
    """Text inside a symbol belongs to it; text-bearing symbols with none inside
    take the nearest free text. Returns the texts left over (arrow labels)."""
    free = list(texts)
    for s in symbols:
        inside = [t for t in free if contains_point(s.box, t.center)]
        if inside:
            s.text_boxes = [t.box for t in sorted(inside, key=lambda t: (t.box[1], t.box[0]))]
            free = [t for t in free if t not in inside]
    for s in symbols:
        if s.cls in TEXT_NODES and not s.text_boxes and free:
            near = min(free, key=lambda t: box_distance(s.box, t.box))
            if box_distance(s.box, near.box) < 0.5 * max(s.box[2] - s.box[0], s.box[3] - s.box[1]):
                s.text_boxes = [near.box]
                free.remove(near)
    return [t.box for t in free]


def postprocess(dets: list[Detection], gray: np.ndarray, mask: np.ndarray | None) -> Cleaned:
    h, w = gray.shape[:2]
    side = max(config.ENDPOINT_BOX_MIN_PX, config.ENDPOINT_BOX_FRAC * max(h, w))
    by = lambda c: [d for d in dets if d.cls == c]

    nodes_raw = [d for d in dets if d.cls in NODE_CLASSES]
    nodes = nms(nodes_raw, config.GENERAL_THRESHOLD)
    texts = nms(by(TEXT), config.GENERAL_THRESHOLD)
    heads = nms(by(HEAD), config.GENERAL_THRESHOLD)
    tails = nms(by(TAIL), config.GENERAL_THRESHOLD)
    arrows = dedupe_arrows(by(ARROW), mask)

    symbols = [Symbol(d.cls, d.box, d.conf) for d in nodes]
    free = attach_text(symbols, texts)
    return Cleaned(
        symbols=symbols,
        arrows=attach_endpoints(arrows, heads, tails, gray, mask, side),
        free_texts=free,
        removed={"symbols": len(nodes_raw) - len(nodes), "text": len(by(TEXT)) - len(texts),
                 "arrows": len(by(ARROW)) - len(arrows)},
    )
