"""Test-set evaluation, comparable to the paper (sections IV-B, IV-C).

Two phases, so the slow model runs happen once:

    python -m hdr.eval.run_eval predict --datasets fca fcb fa   # models -> out/eval/cache
    python -m hdr.eval.run_eval score   --datasets fca fcb fa   # metrics from the cache

Metrics
    Symbol detection   per-class AP at IoU 0.8 (paper Tables 4 and 6) for
                       (a) raw YOLOv7 output (conf 0.001, NMS 0.65, like YOLOv7's
                           own test) and
                       (b) after our post-processing (paper section III-C).
                       Also mAP@0.5, and precision / recall at IoU 0.8 after
                       post-processing (paper Table 5).
    Full diagram       paper criterion: same number of symbols as the ground truth,
                       every symbol IoU >= 0.8, every arrow's source and target correct.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from hdr import config
from hdr.data.coco import load_split
from hdr.data.coco_to_yolo import endpoint_box

CACHE = config.OUT_ROOT / "eval" / "cache"
RAW_CONF, RAW_IOU = 0.001, 0.65        # YOLOv7 test.py defaults
PIPE_CONF, PIPE_IOU = 0.25, 0.9        # pipeline settings (hdr.infer.models.Detector)


# ---------------------------------------------------------------- predict

def predict(datasets: list[str], limit: int = 0) -> None:
    import torch
    from hdr.infer.models import Detector, Segmenter, default_weights
    torch.set_num_threads(max(1, torch.get_num_threads()))
    seg = None
    for ds in datasets:
        w = default_weights(ds)
        det = Detector(w["yolo"])
        seg = seg or Segmenter(w["deeplab"])     # one DeepLab for all datasets (paper)
        out = CACHE / ds
        out.mkdir(parents=True, exist_ok=True)
        ims = load_split(ds, "test")[: limit or None]
        t0 = time.time()
        for i, im in enumerate(ims, 1):
            stem = Path(im.file_name).stem
            if (out / f"{stem}.json").exists() and (out / f"{stem}_mask.png").exists():
                continue
            img = cv2.imread(str(im.path))
            raw = det.forward(img)
            dump = lambda ds_: [[d.cls, list(d.box), d.conf] for d in ds_]
            (out / f"{stem}.json").write_text(json.dumps({
                "raw": dump(det.nms(raw, RAW_CONF, RAW_IOU)),
                "pipe": dump(det.nms(raw, PIPE_CONF, PIPE_IOU)),
            }))
            cv2.imwrite(str(out / f"{stem}_mask.png"), seg(img) * 255)
            if i % 10 == 0 or i == len(ims):
                el = time.time() - t0
                print(f"[{ds}] {i}/{len(ims)}  {el / i:.1f}s/image  ~{el / i * (len(ims) - i) / 60:.0f} min left",
                      flush=True)


# ---------------------------------------------------------------- scoring helpers

def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def gt_boxes(im) -> list[tuple[str, tuple]]:
    """Ground-truth boxes incl. arrow_head / arrow_tail (built like the training labels)."""
    out = []
    for a in im.annotations:
        x, y, w, h = a.bbox
        out.append((a.category, (x, y, x + w, y + h)))
        if a.category == "arrow" and a.keypoints:
            for name, k in (("arrow_head", config.ARROW_HEAD_KP_INDEX), ("arrow_tail", config.ARROW_TAIL_KP_INDEX)):
                bx, by, bw, bh = endpoint_box(a.keypoints[3 * k], a.keypoints[3 * k + 1], im.width, im.height)
                out.append((name, (bx, by, bx + bw, by + bh)))
    return out


def average_precision(per_image: list[tuple[list, list]], cls: str, thr: float) -> tuple[float, int, list]:
    """All-point interpolated AP for one class. per_image = [(dets, gts)].
    Returns (AP, number of GT, [(score, is_tp), ...])."""
    scored, n_gt = [], 0
    for dets, gts in per_image:
        g = [b for c, b in gts if c == cls]
        n_gt += len(g)
        used = [False] * len(g)
        for c, b, s in sorted((d for d in dets if d[0] == cls), key=lambda d: -d[2]):
            best, bi = 0.0, -1
            for j, gb in enumerate(g):
                if not used[j]:
                    v = iou(b, gb)
                    if v > best:
                        best, bi = v, j
            tp = best >= thr
            if tp:
                used[bi] = True
            scored.append((s, tp))
    if n_gt == 0:
        return float("nan"), 0, scored
    scored.sort(key=lambda x: -x[0])
    tp = np.cumsum([t for _, t in scored]) if scored else np.array([0])
    fp = np.cumsum([not t for _, t in scored]) if scored else np.array([0])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), n_gt, scored


def postprocessed_dets(pipe: list, gray: np.ndarray, mask: np.ndarray):
    """Run our post-processing on cached pipeline detections; return (dets for AP, cleaned)."""
    from hdr.infer.models import Detection
    from hdr.postprocess import TEXT, nms, postprocess
    dets = [Detection(c, tuple(b), s) for c, b, s in pipe]
    cleaned = postprocess(dets, gray, mask)
    out = [[s.cls, list(s.box), s.conf] for s in cleaned.symbols]
    out += [["text", list(t.box), t.conf] for t in nms([d for d in dets if d.cls == TEXT], config.GENERAL_THRESHOLD)]
    for a in cleaned.arrows:
        out.append(["arrow", list(a.box), a.conf])
        if a.head:
            out.append(["arrow_head", list(a.head), a.head_conf])
        if a.tail:
            out.append(["arrow_tail", list(a.tail), a.tail_conf])
    return out, cleaned


# ---------------------------------------------------------------- score

def score(datasets: list[str]) -> dict:
    from hdr.eval.graph_check import score as graph_score
    from hdr.graph import build_graph
    results = {}
    for ds in datasets:
        ims = [im for im in load_split(ds, "test") if (CACHE / ds / f"{Path(im.file_name).stem}.json").exists()]
        raw_pi, post_pi, recog, edges_ok, edges_gt = [], [], 0, 0, 0
        for im in ims:
            stem = Path(im.file_name).stem
            c = json.loads((CACHE / ds / f"{stem}.json").read_text())
            gts = gt_boxes(im)
            gray = cv2.imread(str(im.path), cv2.IMREAD_GRAYSCALE)
            mask = (cv2.imread(str(CACHE / ds / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            post, cleaned = postprocessed_dets(c["pipe"], gray, mask)
            raw_pi.append((c["raw"], gts))
            post_pi.append((post, gts))
            side = max(config.ENDPOINT_BOX_MIN_PX, config.ENDPOINT_BOX_FRAC * max(gray.shape))
            s = graph_score(build_graph(cleaned, [], 1.5 * side, (im.width, im.height), allow_self_loops=ds == "fa"), im)
            recog += s.recognized
            edges_ok += s.correct_edges
            edges_gt += s.gt_edges
        classes = config.DATASET_CLASSES[ds]
        per_class = {}
        for cls in classes:
            ap_raw, n, _ = average_precision(raw_pi, cls, 0.8)
            ap_post, _, sc = average_precision(post_pi, cls, 0.8)
            ap50, _, _ = average_precision(post_pi, cls, 0.5)
            tp = sum(t for _, t in sc)
            per_class[cls] = {"gt": n, "ap80_raw": ap_raw, "ap80_post": ap_post, "ap50_post": ap50,
                              "precision80_post": tp / max(1, len(sc)), "recall80_post": tp / max(1, n)}
        mean = lambda k: float(np.nanmean([v[k] for v in per_class.values()]))
        results[ds] = {"images": len(ims), "per_class": per_class,
                       "map80_raw": mean("ap80_raw"), "map80_post": mean("ap80_post"),
                       "map50_post": mean("ap50_post"), "precision80_post": mean("precision80_post"),
                       "recall80_post": mean("recall80_post"),
                       "diagram_recognition": recog / max(1, len(ims)), "recognized": recog,
                       "edge_accuracy": edges_ok / max(1, edges_gt)}
    return results


def print_results(results: dict) -> None:
    for ds, r in results.items():
        print(f"\n=== {ds.upper()} test ({r['images']} images) ===")
        print(f"{'class':12} {'GT':>5} {'AP80 raw':>9} {'AP80 post':>10} {'P80':>6} {'R80':>6}")
        for cls, v in r["per_class"].items():
            print(f"{cls:12} {v['gt']:5d} {v['ap80_raw'] * 100:8.1f}% {v['ap80_post'] * 100:9.1f}% "
                  f"{v['precision80_post'] * 100:5.1f}% {v['recall80_post'] * 100:5.1f}%")
        print(f"{'mean':12} {'':5} {r['map80_raw'] * 100:8.1f}% {r['map80_post'] * 100:9.1f}% "
              f"{r['precision80_post'] * 100:5.1f}% {r['recall80_post'] * 100:5.1f}%   (mAP@0.5 post {r['map50_post'] * 100:.1f}%)")
        print(f"full diagram recognition: {r['recognized']}/{r['images']} = {r['diagram_recognition'] * 100:.1f}%   "
              f"(arrows linked correctly: {r['edge_accuracy'] * 100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "score"])
    ap.add_argument("--datasets", nargs="+", default=["fca", "fcb", "fa"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "predict":
        predict(args.datasets, args.limit)
    else:
        res = score(args.datasets)
        print_results(res)
        (config.OUT_ROOT / "eval" / "results.json").write_text(json.dumps(res, indent=2))
        print(f"\nsaved {config.OUT_ROOT / 'eval' / 'results.json'}")


if __name__ == "__main__":
    main()
