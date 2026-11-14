from pathlib import Path

import pytest

from hdr import config
from hdr.data.check_conventions import box_dist
from hdr.data.coco import Annotation, ImageRecord, load_split
from hdr.data.coco_to_yolo import clip_box, endpoint_box, image_labels, to_yolo_line


def test_endpoint_box_is_centred_square():
    x, y, w, h = endpoint_box(500, 500, 1000, 800, frac=0.03, min_px=16)
    assert w == pytest.approx(30) and h == pytest.approx(30)
    assert x + w / 2 == pytest.approx(500) and y + h / 2 == pytest.approx(500)


def test_endpoint_box_min_size_and_clipping():
    x, y, w, h = endpoint_box(2, 3, 100, 100, frac=0.03, min_px=16)
    assert (x, y) == (0.0, 0.0)
    assert w == pytest.approx(10) and h == pytest.approx(11)


def test_clip_box_outside_gives_zero_area():
    assert clip_box((120, 10, 5, 5), 100, 100)[2] == 0


def test_to_yolo_line_normalises():
    assert to_yolo_line(3, (0, 0, 50, 100), 100, 200) == "3 0.250000 0.250000 0.500000 0.500000"


def test_image_labels_adds_head_and_tail():
    arrow = Annotation(id=1, aid=None, category="arrow", bbox=(10, 10, 20, 80),
                       keypoints=[20.0, 10.0, 2.0, 20.0, 90.0, 2.0])
    im = ImageRecord(id=0, file_name="x.png", path=Path("x.png"), width=100, height=100,
                     annotations=[arrow])
    classes = config.FLOWCHART_CLASSES
    lines = [l.split() for l in image_labels(im, classes)]
    by_cls = {classes[int(l[0])]: l for l in lines}
    assert set(by_cls) == {"arrow", "arrow_head", "arrow_tail"}
    # keypoint 1 (y=90) is the head, keypoint 0 (y=10) the tail
    assert float(by_cls["arrow_head"][2]) > float(by_cls["arrow_tail"][2])


def test_image_labels_rejects_unknown_category():
    bad = Annotation(id=1, aid=None, category="cloud", bbox=(0, 0, 5, 5))
    im = ImageRecord(id=0, file_name="x.png", path=Path("x.png"), width=10, height=10,
                     annotations=[bad])
    with pytest.raises(KeyError):
        image_labels(im, config.FLOWCHART_CLASSES)


@pytest.mark.skipif(not (config.DATASETS_ROOT / "fcb" / "val.json").exists(),
                    reason="dataset not present")
def test_head_keypoint_lies_at_arrow_next_on_real_data():
    hits = total = 0
    for im in load_split("fcb", "val")[:30]:
        by_id = {a.id: a for a in im.annotations}
        for a in im.annotations:
            nxt = by_id.get(a.arrow_next)
            if a.category != "arrow" or nxt is None:
                continue
            h = config.ARROW_HEAD_KP_INDEX
            t = config.ARROW_TAIL_KP_INDEX
            head = a.keypoints[3 * h:3 * h + 2]
            tail = a.keypoints[3 * t:3 * t + 2]
            total += 1
            hits += box_dist(head, nxt.bbox) < box_dist(tail, nxt.bbox)
    assert total > 100 and hits / total > 0.99
