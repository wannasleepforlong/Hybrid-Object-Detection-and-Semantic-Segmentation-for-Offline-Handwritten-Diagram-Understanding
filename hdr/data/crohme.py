"""Render CROHME InkML math expressions to images for TrOCR (paper section III-D).

CROHME is not bundled. Download a CROHME release (InkML files with a
`truth` annotation) and pass the folder; it is searched recursively.

Only simple linear expressions are kept: the flowchart text uses arithmetic
operators (+, -, *, /, =, <, >) on one line, not fractions, roots or
sub/superscripts. LaTeX is turned into plain text (\\times -> *, \\div -> /).

Output: data_work/text/crohme/<name>.png and data_work/text/crohme.tsv
Usage: python -m hdr.data.crohme --src path/to/CROHME
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from hdr import config

NS = {"ink": "http://www.w3.org/2003/InkML"}

LATEX_MAP = [
    (r"\times", "*"), (r"\div", "/"), (r"\cdot", "*"), (r"\lt", "<"), (r"\gt", ">"),
    (r"\leq", "<="), (r"\geq", ">="), (r"\neq", "!="), (r"\pm", "+-"),
    (r"\{", "("), (r"\}", ")"), (r"\left", ""), (r"\right", ""),
]
REJECT = re.compile(r"\\(frac|sqrt|sum|int|lim|log|sin|cos|tan|infty|alpha|beta|gamma|theta|pi|"
                    r"sigma|mu|lambda|phi|Delta|ldots|rightarrow|exists|forall|in)|[\^_]")
ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/=<>()[]., !")


def latex_to_plain(latex: str) -> str | None:
    s = latex.strip().strip("$").strip()
    if REJECT.search(s):
        return None
    for src, dst in LATEX_MAP:
        s = s.replace(src, dst)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    if not s or not set(s) <= ALLOWED or "\\" in s:
        return None
    return s


def parse_inkml(path: Path) -> tuple[str | None, list[np.ndarray]]:
    root = ET.parse(path).getroot()
    truth = None
    for ann in root.findall("ink:annotation", NS):
        if ann.get("type") == "truth":
            truth = ann.text
    traces = []
    for tr in root.findall("ink:trace", NS):
        pts = []
        for p in (tr.text or "").strip().split(","):
            vals = p.split()
            if len(vals) >= 2:
                pts.append((float(vals[0]), float(vals[1])))
        if pts:
            traces.append(np.asarray(pts))
    return truth, traces


def render_traces(traces: list[np.ndarray], height: int = 96, thickness: int = 3) -> np.ndarray:
    allp = np.vstack(traces)
    mn, mx = allp.min(0), allp.max(0)
    span = max(mx[1] - mn[1], 1e-6)
    scale = (height - 2 * 12) / span
    w = int((mx[0] - mn[0]) * scale) + 24
    # very flat expressions (e.g. "-") would get a huge scale; cap the width
    if w > 2000:
        scale *= 2000 / w
        w = 2000
    img = np.full((height, max(w, 24)), 255, np.uint8)
    for t in traces:
        p = ((t - mn) * scale + 12).astype(np.int32)
        cv2.polylines(img, [p.reshape(-1, 1, 2)], False, 0, thickness, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=config.TEXT_ROOT / "crohme")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, seen, kept = [], 0, 0
    for f in sorted(args.src.rglob("*.inkml")):
        seen += 1
        try:
            truth, traces = parse_inkml(f)
        except ET.ParseError:
            continue
        text = latex_to_plain(truth) if truth else None
        if not text or not traces:
            continue
        name = f"crohme_{kept:06d}.png"
        cv2.imwrite(str(args.out / name), render_traces(traces))
        rows.append(f"{args.out.name}/{name}\t{text}")
        kept += 1
    (args.out.parent / f"{args.out.name}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"kept {kept} of {seen} InkML files -> {args.out}")


if __name__ == "__main__":
    main()
