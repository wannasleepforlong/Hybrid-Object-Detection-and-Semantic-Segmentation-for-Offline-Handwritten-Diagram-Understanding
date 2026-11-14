"""Directed graph from cleaned detections, plus Mermaid export (paper section III-E).

Edge rule (paper): for each arrow, the source is the symbol whose box overlaps
the arrow's TAIL box most, the target is the symbol overlapping its HEAD box
most. Overlap, not distance, so an arrow passing near an unrelated symbol is
not linked to it. When no symbol overlaps an endpoint at all, the nearest
symbol within a small radius is used and the edge is flagged.

Arrow labels (yes/no on decision branches, transition symbols in automata) are
free text boxes, each given to the nearest arrow.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from hdr.postprocess import Box, Cleaned, box_distance, center, iou

MERMAID_SHAPES = {
    # class: (open, close)
    "terminator": ('(["', '"])'),
    "process": ('["', '"]'),
    "decision": ('{"', '"}'),
    "data": ('[/"', '"/]'),
    "connection": ('(("', '"))'),
    "state": ('(("', '"))'),
    "final state": ('((("', '")))'),
}


@dataclass
class Node:
    id: str
    cls: str
    box: Box
    text: str = ""
    conf: float = 1.0


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    arrow: int = -1            # index into Cleaned.arrows
    matched_by: str = "iou"    # "iou" or "nearest" (fallback)
    tail: tuple[float, float] | None = None   # arrow start point in the image
    head: tuple[float, float] | None = None   # arrow end point in the image
    endpoints: str = ""        # "detected" / "skeleton" for head and tail
    conf: float = 1.0          # arrow detection confidence


@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]
    dangling: list[int] = field(default_factory=list)   # arrows with an unmatched end
    size: tuple[int, int] = (0, 0)                      # image width, height
    merged: list[int] = field(default_factory=list)     # duplicate arrows dropped

    def to_json(self) -> str:
        return json.dumps({"image_size": self.size,
                           "nodes": [asdict(n) for n in self.nodes],
                           "edges": [asdict(e) for e in self.edges],
                           "dangling_arrows": self.dangling,
                           "merged_duplicate_arrows": self.merged}, indent=2)

    def to_mermaid(self, direction: str | None = None) -> str:
        from hdr.export import to_mermaid
        return to_mermaid(self, direction)


def load_graph(text: str) -> Graph:
    """Inverse of Graph.to_json (e.g. after fixing a misread text by hand)."""
    d = json.loads(text)
    tup = lambda v: tuple(v) if v is not None else None
    nodes = [Node(n["id"], n["cls"], tuple(n["box"]), n.get("text", ""), n.get("conf", 1.0)) for n in d["nodes"]]
    edges = [Edge(**{**e, "tail": tup(e.get("tail")), "head": tup(e.get("head"))}) for e in d["edges"]]
    return Graph(nodes, edges, d.get("dangling_arrows", []), tuple(d.get("image_size", (0, 0))),
                 d.get("merged_duplicate_arrows", []))


def match_endpoint(end: Box, nodes: list[Node], radius: float) -> tuple[str | None, str]:
    """Node overlapping the endpoint box most; else the nearest within radius."""
    if not nodes:
        return None, ""
    scores = [iou(end, n.box) for n in nodes]
    best = int(np.argmax(scores))
    if scores[best] > 0:
        return nodes[best].id, "iou"
    dists = [box_distance(end, n.box) for n in nodes]
    best = int(np.argmin(dists))
    return (nodes[best].id, "nearest") if dists[best] <= radius else (None, "")


def _point_segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def reading_order(symbols) -> list:
    """Top-to-bottom, then left-to-right (rows = about half a typical symbol height)."""
    if not symbols:
        return []
    row = max(1.0, float(np.median([s.box[3] - s.box[1] for s in symbols])) / 2)
    return sorted(symbols, key=lambda s: (round(center(s.box)[1] / row), center(s.box)[0]))


def build_graph(cleaned: Cleaned, free_text: list[str], radius: float,
                size: tuple[int, int] = (0, 0), allow_self_loops: bool = False) -> Graph:
    """allow_self_loops: True for automata (a state can loop to itself); in a
    flowchart an arrow never starts and ends at the same symbol, so such a link
    means one endpoint was misplaced and that end is re-linked."""
    ordered = reading_order(cleaned.symbols)
    nodes = [Node(f"n{i}", s.cls, s.box, s.text, s.conf) for i, s in enumerate(ordered)]
    by_id = {n.id: n for n in nodes}
    edges, dangling = [], []
    for k, a in enumerate(cleaned.arrows):
        src, how_s = match_endpoint(a.tail, nodes, radius) if a.tail else (None, "")
        dst, how_d = match_endpoint(a.head, nodes, radius) if a.head else (None, "")
        if src is not None and src == dst and not allow_self_loops:
            # the end that overlaps the shared symbol less is the misplaced one
            others = [n for n in nodes if n.id != src]
            if iou(a.head, by_id[src].box) <= iou(a.tail, by_id[src].box):
                dst, how_d = match_endpoint(a.head, others, 3 * radius)
            else:
                src, how_s = match_endpoint(a.tail, others, 3 * radius)
            how_s = how_d = "relinked" if src and dst else ""
        if src is None or dst is None:
            dangling.append(k)
            continue
        edges.append(Edge(src, dst, arrow=k,
                          matched_by=("iou" if how_s == how_d == "iou"
                                      else "relinked" if "relinked" in (how_s, how_d) else "nearest"),
                          tail=center(a.tail), head=center(a.head), conf=a.conf,
                          endpoints=f"tail {a.tail_source or '-'}, head {a.head_source or '-'}"))

    # Two arrows linking the same pair the same way: a duplicate detection that
    # survived NMS and the mask check. Keep the more confident arrow.
    best: dict[tuple[str, str], Edge] = {}
    for e in edges:
        key = (e.src, e.dst)
        if key not in best or e.conf > best[key].conf:
            best[key] = e
    merged = [e.arrow for e in edges if best[(e.src, e.dst)] is not e]
    edges = [e for e in edges if best[(e.src, e.dst)] is e]

    # labels: each free text goes to the nearest arrow (its tail-to-head segment)
    for box, text in zip(cleaned.free_texts, free_text):
        if not text or not edges:
            continue
        c = center(box)
        best = min(edges, key=lambda e: _point_segment_distance(
            c, center(cleaned.arrows[e.arrow].tail), center(cleaned.arrows[e.arrow].head)))
        a = cleaned.arrows[best.arrow]
        reach = max(a.box[2] - a.box[0], a.box[3] - a.box[1]) * 0.75 + 40
        if _point_segment_distance(c, center(a.tail), center(a.head)) <= reach:
            best.label = f"{best.label} {text}".strip()
    return Graph(nodes, edges, dangling, size, merged)


def to_mermaid(g: Graph, direction: str | None = "TD") -> str:
    """Kept for compatibility; see hdr.export.to_mermaid."""
    from hdr.export import to_mermaid as _to_mermaid
    return _to_mermaid(g, direction)
