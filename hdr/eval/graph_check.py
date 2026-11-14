"""Compare a predicted graph with the dataset's ground truth for one image.

Nodes are matched one-to-one by box overlap (IoU >= 0.5, best first). An edge
is correct when its two matched nodes are linked by a ground-truth arrow
(arrow_prev -> arrow_next). The paper's "diagram recognized" criterion
(section IV-C): same number of symbols, every symbol IoU >= 0.8 with the
ground truth, and every arrow's predecessor and successor correct.
"""
from __future__ import annotations

from dataclasses import dataclass

from hdr.data.coco import ImageRecord
from hdr.graph import Graph
from hdr.postprocess import NODE_CLASSES, iou


@dataclass
class GraphScore:
    gt_nodes: int
    pred_nodes: int
    matched_nodes: int
    class_correct: int
    gt_edges: int
    pred_edges: int
    correct_edges: int
    min_matched_iou: float
    recognized: bool

    def summary(self) -> str:
        return (f"nodes {self.matched_nodes}/{self.gt_nodes} matched (pred {self.pred_nodes}), "
                f"class ok {self.class_correct}/{self.matched_nodes}, "
                f"edges {self.correct_edges}/{self.gt_edges} correct (pred {self.pred_edges}), "
                f"min IoU {self.min_matched_iou:.2f}, "
                f"diagram recognized (paper criterion): {'YES' if self.recognized else 'no'}")


def gt_graph(im: ImageRecord) -> tuple[dict[int, tuple], set[tuple[int, int]]]:
    nodes = {a.id: (a.category, (a.bbox[0], a.bbox[1], a.bbox[0] + a.bbox[2], a.bbox[1] + a.bbox[3]))
             for a in im.annotations if a.category in NODE_CLASSES}
    edges = {(a.arrow_prev, a.arrow_next) for a in im.annotations
             if a.category == "arrow" and a.arrow_prev in nodes and a.arrow_next in nodes}
    return nodes, edges


def score(g: Graph, im: ImageRecord) -> GraphScore:
    gt_nodes, gt_edges = gt_graph(im)
    pairs = sorted(((iou(n.box, gb), n.id, gid) for n in g.nodes for gid, (_, gb) in gt_nodes.items()),
                   reverse=True)
    match, used, ious = {}, set(), []
    for v, nid, gid in pairs:
        if v < 0.5:
            break
        if nid not in match and gid not in used:
            match[nid] = gid
            used.add(gid)
            ious.append(v)
    by_id = {n.id: n for n in g.nodes}
    class_ok = sum(by_id[n].cls == gt_nodes[gid][0] for n, gid in match.items())
    correct = sum((match.get(e.src), match.get(e.dst)) in gt_edges for e in g.edges)
    recognized = (len(g.nodes) == len(gt_nodes) and len(match) == len(gt_nodes)
                  and min(ious, default=0) >= 0.8 and class_ok == len(gt_nodes)
                  and correct == len(gt_edges) == len(g.edges))
    return GraphScore(len(gt_nodes), len(g.nodes), len(match), class_ok, len(gt_edges),
                      len(g.edges), correct, min(ious, default=0.0), recognized)
