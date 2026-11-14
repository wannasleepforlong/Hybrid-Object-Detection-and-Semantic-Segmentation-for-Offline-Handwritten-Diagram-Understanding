import numpy as np
import pytest

from hdr import config
from hdr.graph import Edge, Graph, Node, build_graph, to_mermaid
from hdr.infer.models import Detection
from hdr.postprocess import (Arrow, Cleaned, Symbol, attach_endpoints, attach_text, dedupe_arrows,
                             iou, nms, postprocess, square)


def D(cls, box, conf=0.9):
    return Detection(cls, tuple(float(v) for v in box), conf)


# ---------------------------------------------------------------- post-processing

def test_nms_same_class_only():
    dets = [D("process", (0, 0, 100, 50), 0.9), D("process", (5, 0, 105, 50), 0.6),
            D("decision", (5, 0, 105, 50), 0.7)]
    kept = nms(dets, 0.5)
    assert [d.conf for d in kept] == [0.9, 0.7]   # same-class duplicate gone, other class kept


def test_dedupe_arrows_uses_mask_components():
    mask = np.zeros((200, 200), np.uint8)
    mask[20:180, 50:53] = 1                        # one vertical arrow stroke
    mask[20:180, 120:123] = 1                      # a second, separate stroke
    same = [D("arrow", (40, 10, 70, 120), 0.9), D("arrow", (40, 60, 70, 190), 0.8)]  # IoU ~0.2
    assert len(dedupe_arrows(same, mask)) == 1     # both cover stroke 1 -> duplicate
    two = [D("arrow", (40, 10, 90, 190), 0.9), D("arrow", (85, 10, 130, 190), 0.8)]
    assert len(dedupe_arrows(two, mask)) == 2      # overlap, but different strokes


def test_attach_endpoints_detected():
    gray = np.full((200, 200), 255, np.uint8)
    a = D("arrow", (40, 20, 60, 180))
    head, tail = D("arrow_head", (40, 165, 60, 185)), D("arrow_tail", (40, 15, 60, 35))
    (arrow,) = attach_endpoints([a], [head], [tail], gray, None, side=20)
    assert arrow.head == head.box and arrow.tail == tail.box
    assert arrow.head_source == arrow.tail_source == "detected"


def test_attach_endpoints_skeleton_fallback_finds_missing_tail():
    gray = np.full((200, 200), 255, np.uint8)
    gray[20:180, 49:52] = 0                        # vertical shaft
    a = D("arrow", (40, 15, 60, 185))
    head = D("arrow_head", (40, 165, 60, 185))     # detected head at the bottom
    (arrow,) = attach_endpoints([a], [head], [], gray, None, side=20)
    assert arrow.tail_source == "skeleton"
    assert (arrow.tail[1] + arrow.tail[3]) / 2 < 40   # tail placed at the top end


def test_attach_text_inside_then_nearest():
    syms = [Symbol("process", (0, 0, 100, 50), 0.9), Symbol("decision", (0, 100, 100, 160), 0.9)]
    texts = [D("text", (10, 10, 90, 40)), D("text", (105, 120, 130, 140)), D("text", (300, 300, 320, 310))]
    free = attach_text(syms, texts)
    assert syms[0].text_boxes == [texts[0].box]
    assert syms[1].text_boxes == [texts[1].box]    # nearest, just outside the diamond
    assert free == [texts[2].box]                  # far text stays a free label


# ---------------------------------------------------------------- graph + mermaid

def _two_boxes_one_arrow():
    syms = [Symbol("terminator", (0, 0, 100, 40), 0.9, text="Start"),
            Symbol("process", (0, 140, 100, 180), 0.9, text="x = 1")]
    arrow = Arrow((40, 40, 60, 140), 0.9, head=square((50, 140), 20), tail=square((50, 40), 20))
    return Cleaned(syms, [arrow], [], {})


def test_build_graph_links_tail_to_head():
    g = build_graph(_two_boxes_one_arrow(), [], radius=30)
    assert [(e.src, e.dst, e.matched_by) for e in g.edges] == [("n0", "n1", "iou")]


def test_build_graph_flags_dangling_arrow():
    c = _two_boxes_one_arrow()
    c.arrows[0].head = square((500, 500), 20)      # head far from every symbol
    g = build_graph(c, [], radius=30)
    assert g.edges == [] and g.dangling == [0]


def test_label_goes_to_nearest_arrow():
    c = _two_boxes_one_arrow()
    c.free_texts = [(65, 80, 90, 95)]
    g = build_graph(c, ["Yes"], radius=30)
    assert g.edges[0].label == "Yes"


def test_mermaid_shapes_and_escaping():
    g = Graph([Node("n0", "terminator", (0, 0, 1, 1), "Start"),
               Node("n1", "decision", (0, 0, 1, 1), 'x > "0"'),
               Node("n2", "data", (0, 0, 1, 1), "read n")],
              [Edge("n0", "n1"), Edge("n1", "n2", label="Yes")])
    m = to_mermaid(g)
    assert m.startswith("flowchart TD")
    assert 'n0(["Start"])' in m and 'n1{"x > #quot;0#quot;"}' in m and 'n2[/"read n"/]' in m
    assert 'n1 -->|"Yes"| n2' in m and "n0 --> n1" in m


# ---------------------------------------------------------------- paper's matching rule on ground truth

def _gt_cleaned(im):
    """Perfect detections from the annotations: true boxes, true head/tail points."""
    side = max(config.ENDPOINT_BOX_MIN_PX, config.ENDPOINT_BOX_FRAC * max(im.width, im.height))
    xyxy = lambda b: (b[0], b[1], b[0] + b[2], b[1] + b[3])
    syms = [Symbol(a.category, xyxy(a.bbox), 1.0) for a in im.annotations
            if a.category in ("connection", "data", "decision", "process", "terminator")]
    hi, ti = config.ARROW_HEAD_KP_INDEX, config.ARROW_TAIL_KP_INDEX
    arrows = [Arrow(xyxy(a.bbox), 1.0,
                    head=square((a.keypoints[3 * hi], a.keypoints[3 * hi + 1]), side),
                    tail=square((a.keypoints[3 * ti], a.keypoints[3 * ti + 1]), side))
              for a in im.annotations if a.category == "arrow"]
    return Cleaned(syms, arrows, [], {}), side


@pytest.mark.skipif(not (config.DATASETS_ROOT / "fcb" / "val.json").exists(), reason="dataset not present")
def test_iou_matching_rule_on_ground_truth_fcb_val():
    from hdr.data.coco import load_split
    from hdr.eval.graph_check import score
    correct = total = 0
    for im in load_split("fcb", "val"):
        cleaned, side = _gt_cleaned(im)
        s = score(build_graph(cleaned, [], radius=1.5 * side), im)
        correct += s.correct_edges
        total += s.gt_edges
    assert total > 1000
    assert correct / total > 0.98, f"{correct}/{total}"


def test_self_link_is_relinked_in_flowcharts_but_kept_for_automata():
    syms = [Symbol("process", (0, 0, 100, 40), 0.9), Symbol("process", (0, 140, 100, 180), 0.9)]
    # head detected wrongly near the source box; its true target is the box below
    arrow = Arrow((40, 30, 60, 150), 0.9, head=square((50, 45), 20), tail=square((50, 35), 20))
    c = Cleaned(syms, [arrow], [], {})
    g = build_graph(c, [], radius=30)
    assert [(e.src, e.dst, e.matched_by) for e in g.edges] == [("n0", "n1", "relinked")]
    g_fa = build_graph(c, [], radius=30, allow_self_loops=True)
    assert [(e.src, e.dst) for e in g_fa.edges] == [("n0", "n0")]


def test_duplicate_arrows_between_same_pair_are_merged():
    c = _two_boxes_one_arrow()
    a = c.arrows[0]
    c.arrows.append(Arrow(a.box, 0.5, head=a.head, tail=a.tail))   # weaker duplicate
    g = build_graph(c, [], radius=30)
    assert len(g.edges) == 1 and g.edges[0].conf == 0.9 and g.merged == [1]


def _two_parallel_strokes():
    """Two arrows drawn in opposite directions side by side (A->B left, B->A right)."""
    mask = np.zeros((220, 220), np.uint8)
    mask[20:200, 98:101] = 1
    mask[20:200, 118:121] = 1
    gray = np.full((220, 220), 255, np.uint8)
    gray[mask > 0] = 0
    return mask, gray


def test_opposite_arrows_do_not_borrow_each_others_ends():
    from hdr.postprocess import assign_ends, stroke_labels
    mask, _ = _two_parallel_strokes()
    down = D("arrow", (85, 15, 112, 205), 0.9)        # left stroke points down
    up = D("arrow", (106, 15, 133, 205), 0.9)         # right stroke points up
    heads = [D("arrow_head", (89, 190, 109, 210), 0.9), D("arrow_head", (109, 10, 129, 30), 0.9)]
    tails = [D("arrow_tail", (89, 10, 109, 30), 0.9), D("arrow_tail", (109, 190, 129, 210), 0.9)]
    head_of, tail_of = assign_ends([down, up], heads, tails, stroke_labels(mask), side=20)
    assert head_of[0] is heads[0] and tail_of[0] is tails[0]     # left: top -> bottom
    assert head_of[1] is heads[1] and tail_of[1] is tails[1]     # right: bottom -> top


def test_long_arc_over_short_arrow_is_not_a_duplicate():
    mask = np.zeros((300, 300), np.uint8)
    mask[100:140, 100:103] = 1                         # short arrow
    mask[20:24, 20:280] = 1                            # long arc, separate stroke...
    mask[100:140, 100:103] = 1
    big = D("arrow", (10, 10, 290, 290), 0.9)          # ...whose box contains the short arrow
    small = D("arrow", (90, 95, 110, 145), 0.92)
    assert iou(big.box, small.box) < 0.3
    assert len(dedupe_arrows([big, small], mask)) == 2
