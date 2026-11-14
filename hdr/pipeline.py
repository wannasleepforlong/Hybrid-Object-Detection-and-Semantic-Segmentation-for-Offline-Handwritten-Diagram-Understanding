"""Hand-drawn diagram image -> graph -> Mermaid.

    python -m hdr.pipeline IMAGE [--dataset fcb] [--out out/run]

Writes to the output folder:
    graph.json     nodes (class, box, text) and edges (with labels)
    diagram.mmd    Mermaid flowchart
    diagram.drawio diagrams.net file with the sketch's real layout
    redraw.svg     clean redraw in the original layout
    report.html    one-page report: input + detections, redraw, Mermaid, tables
                   (mermaid.js loads from a CDN; the diagram stays in the file)
    overlay.png    detections on the image: symbols, arrows, green tail / red head,
                   node ids matching graph.json and diagram.mmd

If the image belongs to one of the datasets (…/datasets/<ds>/<split>/<file>), the
graph is also scored against the ground truth.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from hdr import config
from hdr.export import report_html, to_drawio, to_mermaid, to_svg
from hdr.graph import Graph, build_graph
from hdr.infer.models import Detector, Reader, Segmenter, default_weights
from hdr.postprocess import Cleaned, center, postprocess

COLORS = {"terminator": (0, 0, 200), "process": (200, 120, 0), "decision": (0, 140, 0),
          "data": (200, 0, 0), "connection": (140, 0, 140), "state": (200, 0, 0),
          "final state": (0, 140, 0)}


def crop(img: np.ndarray, box, pad: int = 3) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = img.shape[:2]
    return img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def read_texts(reader: Reader, img: np.ndarray, cleaned: Cleaned) -> list[str]:
    """OCR every text box once: symbol texts are filled in place, free texts returned."""
    boxes = [b for s in cleaned.symbols for b in s.text_boxes] + list(cleaned.free_texts)
    texts = reader([crop(img, b) for b in boxes]) if boxes else []
    i = 0
    for s in cleaned.symbols:
        s.text = " ".join(texts[i:i + len(s.text_boxes)]).strip()
        i += len(s.text_boxes)
    return texts[i:]


def draw_overlay(img: np.ndarray, cleaned: Cleaned, g: Graph) -> np.ndarray:
    out = img.copy()
    for n in g.nodes:
        x1, y1, x2, y2 = map(int, n.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS.get(n.cls, (90, 90, 90)), 2)
        cv2.putText(out, n.id, (x1 + 3, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    for b in cleaned.free_texts:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), (150, 150, 150), 1)
    for a in cleaned.arrows:
        x1, y1, x2, y2 = map(int, a.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 255), 1)
        for b, col in ((a.tail, (0, 200, 0)), (a.head, (0, 0, 255))):
            if b:
                cv2.circle(out, tuple(map(int, center(b))), 6, col, -1)
    return out




def dataset_record(image: Path):
    """(dataset, ImageRecord) if the image is a dataset image, else None."""
    from hdr.data.coco import load_split
    split, ds = image.parent.name, image.parent.parent.name
    if ds not in config.DATASET_CLASSES or split not in ("train", "val", "test"):
        return None
    try:
        rec = next((r for r in load_split(ds, split) if r.file_name == image.name), None)
    except FileNotFoundError:
        return None
    return (ds, rec) if rec else None


def run(image: Path, dataset: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image))
    if img is None:
        raise SystemExit(f"cannot read image: {image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = default_weights(dataset)
    t = {}

    t0 = time.time(); dets = Detector(w["yolo"])(img); t["detect"] = time.time() - t0
    t0 = time.time(); mask = Segmenter(w["deeplab"])(img); t["segment"] = time.time() - t0
    cleaned = postprocess(dets, gray, mask)
    t0 = time.time(); free_text = read_texts(Reader(w["trocr"]), img, cleaned); t["ocr"] = time.time() - t0

    side = max(config.ENDPOINT_BOX_MIN_PX, config.ENDPOINT_BOX_FRAC * max(gray.shape))
    g = build_graph(cleaned, free_text, radius=1.5 * side, size=(img.shape[1], img.shape[0]),
                    allow_self_loops=dataset == "fa")
    mermaid = to_mermaid(g)   # direction picked from the drawing

    rec = dataset_record(image)
    score = None
    if rec:
        from hdr.eval.graph_check import score as graph_score
        score = graph_score(g, rec[1])

    files = {"graph.json": "graph.json", "diagram.mmd": "diagram.mmd",
             "diagram.drawio": "diagram.drawio", "redraw.svg": "redraw.svg"}
    (out / "graph.json").write_text(g.to_json(), encoding="utf-8")
    (out / "diagram.mmd").write_text(mermaid, encoding="utf-8")
    (out / "diagram.drawio").write_text(to_drawio(g, image.stem), encoding="utf-8")
    (out / "redraw.svg").write_text(to_svg(g), encoding="utf-8")
    (out / "report.html").write_text(
        report_html(g, cleaned, img, image.name, mermaid, {"time": t}, score, files), encoding="utf-8")
    cv2.imwrite(str(out / "overlay.png"), draw_overlay(img, cleaned, g))

    print(f"{image.name}: {len(dets)} raw detections -> {len(g.nodes)} symbols, "
          f"{len(cleaned.arrows)} arrows, {len(g.edges)} edges, {len(g.dangling)} dangling arrows")
    print("duplicates removed:", cleaned.removed)
    print("endpoints from skeleton fallback:",
          sum(a.head_source == "skeleton" for a in cleaned.arrows), "heads,",
          sum(a.tail_source == "skeleton" for a in cleaned.arrows), "tails")
    print("time (s):", {k: round(v, 1) for k, v in t.items()})
    print("\n" + mermaid)

    result = {"graph": g, "cleaned": cleaned}
    if score is not None:
        print("ground truth check:", score.summary())
        result["score"] = score
    print(f"outputs in {out}  (open report.html)")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--dataset", default="fcb", choices=["fca", "fcb", "fa"],
                    help="which trained YOLOv7 to use (fca/fcb: flowcharts, fa: automata)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.image, args.dataset, args.out or config.OUT_ROOT / "pipeline" / args.image.stem)


if __name__ == "__main__":
    main()
