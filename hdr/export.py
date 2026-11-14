"""Export a recognized diagram: Mermaid, draw.io, a clean SVG redraw, and an HTML report.

What each export keeps from the pipeline output:
    Mermaid   structure + text + labels; symbol type as colour; uncertain edges
              (matched by the nearest-symbol fallback) dashed; low-confidence
              symbols outlined; dangling arrows listed as comments. Layout is
              Mermaid's own (it has no positions); direction is chosen from the
              drawing (mostly vertical arrows -> TD, else LR).
    draw.io   everything above plus the REAL layout: every symbol at its position
              and size in the sketch, every arrow leaving and entering at the
              point where it was drawn. Opens and edits in diagrams.net.
    SVG       a clean redraw in the original layout (straight or L-shaped arrows).
    HTML      one self-contained report: the input with interactive detections,
              the clean redraw, the Mermaid diagram, symbol and connection tables
              (hover a row to highlight it), statistics, and the Mermaid source.
"""
from __future__ import annotations

import base64
import html
import json

import cv2
import numpy as np

from hdr.graph import Graph, Node

LOW_CONF = 0.5

TYPE_COLORS = {  # fill, stroke
    "terminator": ("#fde2e1", "#c0392b"),
    "process": ("#dbe9f9", "#2e6fb7"),
    "decision": ("#ddf2dc", "#2e8b3d"),
    "data": ("#fff1cc", "#b7881a"),
    "connection": ("#efe0f7", "#7d3c98"),
    "state": ("#dbe9f9", "#2e6fb7"),
    "final state": ("#ddf2dc", "#2e8b3d"),
}

MERMAID_SHAPES = {
    "terminator": ('(["', '"])'),
    "process": ('["', '"]'),
    "decision": ('{"', '"}'),
    "data": ('[/"', '"/]'),
    "connection": ('(("', '"))'),
    "state": ('(("', '"))'),
    "final state": ('((("', '")))'),
}


# ---------------------------------------------------------------- mermaid

def _mermaid_text(text: str) -> str:
    return text.replace('"', "#quot;").replace("\n", " ").strip()


def layout_direction(g: Graph) -> str:
    """TD if arrows mostly run vertically in the drawing, else LR."""
    dx = sum(abs(e.head[0] - e.tail[0]) for e in g.edges if e.head and e.tail)
    dy = sum(abs(e.head[1] - e.tail[1]) for e in g.edges if e.head and e.tail)
    return "TD" if dy >= dx else "LR"


def to_mermaid(g: Graph, direction: str | None = None) -> str:
    lines = [f"flowchart {direction or layout_direction(g)}"]
    for n in g.nodes:
        open_, close = MERMAID_SHAPES.get(n.cls, ('["', '"]'))
        lines.append(f"    {n.id}{open_}{_mermaid_text(n.text) or ' '}{close}")
    for e in g.edges:
        arrow = "-.->" if e.matched_by != "iou" else "-->"
        label = f'|"{_mermaid_text(e.label)}"|' if e.label else ""
        lines.append(f"    {e.src} {arrow}{label} {e.dst}")
    used = sorted({n.cls for n in g.nodes})
    for cls in used:
        fill, stroke = TYPE_COLORS.get(cls, ("#eeeeee", "#555555"))
        name = cls.replace(" ", "_")
        lines.append(f"    classDef {name} fill:{fill},stroke:{stroke},color:#222")
        lines.append(f"    class {','.join(n.id for n in g.nodes if n.cls == cls)} {name}")
    low = [n.id for n in g.nodes if n.conf < LOW_CONF]
    if low:
        lines.append("    classDef lowconf stroke:#e67e22,stroke-width:3px,stroke-dasharray:6 4")
        lines.append(f"    class {','.join(low)} lowconf")
    if g.dangling:
        lines.append(f"    %% {len(g.dangling)} arrow(s) had an unmatched end and are not drawn")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- draw.io

DRAWIO_STYLES = {
    "terminator": "rounded=1;arcSize=50;",
    "process": "rounded=0;",
    "decision": "rhombus;",
    "data": "shape=parallelogram;perimeter=parallelogramPerimeter;fixedSize=1;size=12;",
    "connection": "ellipse;",
    "state": "ellipse;",
    "final state": "ellipse;shape=doubleEllipse;",
}


def _rel(point, box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(np.clip((point[0] - x1) / max(1, x2 - x1), 0, 1)),
            float(np.clip((point[1] - y1) / max(1, y2 - y1), 0, 1)))


def to_drawio(g: Graph, name: str = "diagram") -> str:
    """diagrams.net file with symbols at their sketched positions."""
    esc = lambda s: html.escape(s, quote=True)
    by_id = {n.id: n for n in g.nodes}
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for n in g.nodes:
        fill, stroke = TYPE_COLORS.get(n.cls, ("#eeeeee", "#555555"))
        style = (DRAWIO_STYLES.get(n.cls, "rounded=0;") + f"whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};" + ("dashed=1;strokeWidth=2;" if n.conf < LOW_CONF else ""))
        x1, y1, x2, y2 = n.box
        cells.append(f'<mxCell id="{n.id}" value="{esc(n.text)}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x1:.0f}" y="{y1:.0f}" width="{x2 - x1:.0f}" height="{y2 - y1:.0f}" '
                     f'as="geometry"/></mxCell>')
    for i, e in enumerate(g.edges):
        style = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;"
        if e.tail and e.head:  # leave / enter where the arrow was drawn
            ex, ey = _rel(e.tail, by_id[e.src].box)
            nx, ny = _rel(e.head, by_id[e.dst].box)
            style += f"exitX={ex:.2f};exitY={ey:.2f};entryX={nx:.2f};entryY={ny:.2f};"
        if e.matched_by != "iou":
            style += "dashed=1;"
        cells.append(f'<mxCell id="e{i}" value="{esc(e.label)}" style="{style}" edge="1" parent="1" '
                     f'source="{e.src}" target="{e.dst}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    w, h = g.size
    return (f'<mxfile host="hdr"><diagram name="{esc(name)}"><mxGraphModel dx="{w}" dy="{h}" grid="0" '
            f'page="0" math="0" shadow="0"><root>{"".join(cells)}</root></mxGraphModel></diagram></mxfile>\n')


# ---------------------------------------------------------------- SVG redraw

def _shape_svg(n: Node) -> str:
    x1, y1, x2, y2 = n.box
    w, h = x2 - x1, y2 - y1
    fill, stroke = TYPE_COLORS.get(n.cls, ("#eeeeee", "#555555"))
    dash = ' stroke-dasharray="7 4"' if n.conf < LOW_CONF else ""
    a = f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"{dash}'
    cx, cy = x1 + w / 2, y1 + h / 2
    if n.cls == "terminator":
        s = f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{h / 2:.1f}" {a}/>'
    elif n.cls == "decision":
        s = f'<polygon points="{cx:.1f},{y1:.1f} {x2:.1f},{cy:.1f} {cx:.1f},{y2:.1f} {x1:.1f},{cy:.1f}" {a}/>'
    elif n.cls == "data":
        k = min(w * 0.15, 25)
        s = f'<polygon points="{x1 + k:.1f},{y1:.1f} {x2:.1f},{y1:.1f} {x2 - k:.1f},{y2:.1f} {x1:.1f},{y2:.1f}" {a}/>'
    elif n.cls in ("connection", "state", "final state"):
        s = f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{w / 2:.1f}" ry="{h / 2:.1f}" {a}/>'
        if n.cls == "final state":
            s += f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{w / 2 - 6:.1f}" ry="{h / 2 - 6:.1f}" fill="none" stroke="{stroke}" stroke-width="2"/>'
    else:
        s = f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" {a}/>'
    size = max(10.0, min(h * 0.32, w / max(1, len(n.text)) * 1.7, 26))
    text = (f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="{size:.0f}" text-anchor="middle" '
            f'dominant-baseline="central">{html.escape(n.text)}</text>') if n.text else ""
    return f'<g class="node" data-id="{n.id}">{s}{text}</g>'


def _crosses(p, q, box, margin=4) -> bool:
    """Does the axis-aligned segment p-q pass through the box interior?"""
    x1, y1, x2, y2 = box[0] + margin, box[1] + margin, box[2] - margin, box[3] - margin
    if p[0] == q[0]:
        return x1 < p[0] < x2 and min(p[1], q[1]) < y2 and max(p[1], q[1]) > y1
    return y1 < p[1] < y2 and min(p[0], q[0]) < x2 and max(p[0], q[0]) > x1


def route(e, by_id) -> list[tuple[float, float]]:
    """Straight arrow if nearly aligned, else an L whose corner avoids other symbols."""
    t, h = e.tail, e.head
    if abs(t[0] - h[0]) < 15:      # nearly vertical: draw it straight
        x = (t[0] + h[0]) / 2
        return [(x, t[1]), (x, h[1])]
    if abs(t[1] - h[1]) < 15:      # nearly horizontal
        y = (t[1] + h[1]) / 2
        return [(t[0], y), (h[0], y)]
    others = [n.box for nid, n in by_id.items() if nid not in (e.src, e.dst)]
    src = by_id[e.src].box
    vertical_first = min(abs(t[1] - src[1]), abs(t[1] - src[3])) <= min(abs(t[0] - src[0]), abs(t[0] - src[2]))
    corners = [(t[0], h[1]), (h[0], t[1])] if vertical_first else [(h[0], t[1]), (t[0], h[1])]
    hits = [sum(_crosses(t, c, b) or _crosses(c, h, b) for b in others) for c in corners]
    corner = corners[int(np.argmin(hits))]
    return [t, corner, h]


def to_svg(g: Graph) -> str:
    w, h = g.size
    by_id = {n.id: n for n in g.nodes}
    fs = max(15, round(max(w, h) * 0.016))          # label size scales with the drawing
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" class="redraw" '
             f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
             '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
             f'<rect width="{w}" height="{h}" fill="white"/>']
    for i, e in enumerate(g.edges):
        if not (e.tail and e.head):
            continue
        pts = route(e, by_id)
        dash = ' stroke-dasharray="8 5"' if e.matched_by != "iou" else ""
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<g class="edge" data-id="e{i}"><polyline points="{d}" fill="none" stroke="#333" '
                     f'stroke-width="2.5" marker-end="url(#ah)"{dash}/>')
        if e.label:
            seg = max(zip(pts, pts[1:]), key=lambda s: np.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]))
            mx, my = (seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2
            lw, lh = 0.62 * fs * len(e.label) + fs, fs * 1.5
            parts.append(f'<rect x="{mx - lw / 2:.1f}" y="{my - lh / 2:.1f}" width="{lw:.1f}" height="{lh:.1f}" '
                         f'rx="5" fill="white" stroke="#999"/><text x="{mx:.1f}" y="{my:.1f}" font-size="{fs}" '
                         f'font-weight="600" text-anchor="middle" dominant-baseline="central">'
                         f'{html.escape(e.label)}</text>')
        parts.append("</g>")
    parts += [_shape_svg(n) for n in g.nodes]
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- HTML report

def _image_data_uri(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def report_html(g: Graph, cleaned, img: np.ndarray, title: str, mermaid: str,
                stats: dict, score=None, files: dict | None = None) -> str:
    data = {
        "size": g.size,
        "nodes": [{"id": n.id, "cls": n.cls, "box": n.box, "text": n.text, "conf": round(n.conf, 3),
                   "color": TYPE_COLORS.get(n.cls, ("#eee", "#555"))[1]} for n in g.nodes],
        "edges": [{"id": f"e{i}", "src": e.src, "dst": e.dst, "label": e.label, "matched_by": e.matched_by,
                   "endpoints": e.endpoints, "conf": round(e.conf, 3), "arrow_box": cleaned.arrows[e.arrow].box,
                   "tail": e.tail, "head": e.head} for i, e in enumerate(g.edges)],
        "dangling": [{"box": cleaned.arrows[k].box} for k in g.dangling],
        "labels": [list(b) for b in cleaned.free_texts],
    }
    chips = [("symbols", len(g.nodes)), ("connections", len(g.edges)),
             ("labels", sum(bool(e.label) for e in g.edges)),
             ("dangling arrows", len(g.dangling)),
             ("fallback links", sum(e.matched_by != "iou" for e in g.edges)),
             ("skeleton endpoints", sum("skeleton" in e.endpoints for e in g.edges)),
             ("low-confidence symbols", sum(n.conf < LOW_CONF for n in g.nodes))]
    chip_html = "".join(f'<span class="chip{" warn" if v and k in ("dangling arrows", "fallback links", "low-confidence symbols") else ""}">'
                        f'<b>{v}</b> {k}</span>' for k, v in chips)
    if score is not None:
        ok = score.recognized
        chip_html += (f'<span class="chip {"good" if ok else "warn"}"><b>{"✓" if ok else "✗"}</b> '
                      f'ground truth: {score.correct_edges}/{score.gt_edges} edges, '
                      f'{score.matched_nodes}/{score.gt_nodes} symbols</span>')
    timing = ", ".join(f"{k} {v:.1f}s" for k, v in stats.get("time", {}).items())
    links = " · ".join(f'<a href="{v}" download>{k}</a>' for k, v in (files or {}).items())
    page = REPORT_TEMPLATE
    for key, val in {
        "__TITLE__": html.escape(title), "__CHIPS__": chip_html, "__TIMING__": html.escape(timing),
        "__LINKS__": links, "__IMG__": _image_data_uri(img), "__SVG__": to_svg(g),
        "__MERMAID__": html.escape(mermaid), "__DATA__": json.dumps(data),
    }.items():
        page = page.replace(key, val)
    return page


def main() -> None:
    """Re-export Mermaid, draw.io and SVG from a (possibly hand-edited) graph.json."""
    import argparse
    from pathlib import Path

    from hdr.graph import load_graph
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("graph_json", type=Path)
    args = ap.parse_args()
    g = load_graph(args.graph_json.read_text(encoding="utf-8"))
    out = args.graph_json.parent
    (out / "diagram.mmd").write_text(to_mermaid(g), encoding="utf-8")
    (out / "diagram.drawio").write_text(to_drawio(g, out.name), encoding="utf-8")
    (out / "redraw.svg").write_text(to_svg(g), encoding="utf-8")
    print(f"re-exported diagram.mmd, diagram.drawio, redraw.svg in {out}")


REPORT_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diagram report</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#667085;--line:#e3e6eb;--accent:#2e6fb7;--warn:#b7791f;--good:#2e8b3d;--hl:#ff3b7f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14171c;--panel:#1d2128;--ink:#e6e8eb;--muted:#9aa3ae;--line:#2c323b;--accent:#6ea8ff}}
:root[data-theme="dark"]{--bg:#14171c;--panel:#1d2128;--ink:#e6e8eb;--muted:#9aa3ae;--line:#2c323b;--accent:#6ea8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px 24px 6px}h1{font-size:18px;margin:0 0 8px}h2{font-size:14px;margin:0 0 8px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}.chip{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:3px 10px}
.chip.warn b{color:var(--warn)}.chip.good b{color:var(--good)}.meta{color:var(--muted);font-size:12px}.meta a{color:var(--accent)}
main{padding:8px 24px 32px;display:grid;gap:16px}.row{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
.view{position:relative;width:100%}.view img,.view svg{display:block;width:100%;height:auto;border-radius:6px}
.view svg.ov{position:absolute;inset:0}.redraw{background:#fff;border-radius:6px}
.mermaid{background:#fff;border-radius:6px;padding:8px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}tr.hl td{background:color-mix(in srgb,var(--hl) 14%,transparent)}
.tag{font-size:11px;padding:1px 7px;border-radius:999px;border:1px solid var(--line);white-space:nowrap}.tag.warn{color:var(--warn);border-color:var(--warn)}
pre{margin:0;padding:10px;background:var(--bg);border-radius:6px;overflow:auto;font-size:12px}
button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:3px 10px;cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--muted);margin-top:8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
.hl-on rect.ovn,.hl-on .node *{transition:opacity .15s}
@media (max-width:600px){header,main{padding-left:16px;padding-right:16px}}
</style></head><body>
<header><h1>__TITLE__</h1><div class="chips">__CHIPS__</div>
<div class="meta">__TIMING__ &nbsp; __LINKS__</div></header>
<main>
<div class="row">
 <div class="panel"><h2>Input + detections</h2><div class="view" id="inview"><img src="__IMG__" alt="input"><svg class="ov" id="ov"></svg></div>
  <div class="legend"><span><i style="background:#2e8b3d"></i>arrow tail</span><span><i style="background:#e0245e"></i>arrow head</span><span><i style="background:#f08c00"></i>arrow box</span><span><i style="background:#999"></i>arrow label</span><span><i style="background:#b00020"></i>dangling arrow</span></div></div>
 <div class="panel"><h2>Clean redraw (same layout)</h2><div class="view" id="redraw">__SVG__</div></div>
 <div class="panel"><h2>Mermaid (auto layout)</h2><pre class="mermaid">__MERMAID__</pre></div>
</div>
<div class="row">
 <div class="panel"><h2>Symbols</h2><table id="nodes"><thead><tr><th>id</th><th>type</th><th>text</th><th>conf.</th></tr></thead><tbody></tbody></table></div>
 <div class="panel"><h2>Connections</h2><table id="edges"><thead><tr><th>from → to</th><th>label</th><th>linked by</th><th>endpoints</th></tr></thead><tbody></tbody></table></div>
</div>
<div class="panel"><h2>Mermaid source <button id="copy">Copy</button></h2><pre id="src">__MERMAID__</pre></div>
</main>
<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const [W, H] = D.size;
const ov = document.getElementById("ov");
ov.setAttribute("viewBox", `0 0 ${W} ${H}`);
const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const rect = (b, attrs) => mk("rect", {x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], fill: "none", ...attrs});
const textOf = id => (D.nodes.find(n => n.id === id) || {}).text || id;
for (const b of D.labels) ov.appendChild(rect(b, {stroke: "#999", "stroke-width": 2, "stroke-dasharray": "4 3"}));
for (const d of D.dangling) ov.appendChild(rect(d.box, {stroke: "#b00020", "stroke-width": 3, "stroke-dasharray": "6 4"}));
for (const e of D.edges) {
  const g = mk("g", {"data-id": e.id, class: "ove"});
  g.appendChild(rect(e.arrow_box, {stroke: "#f08c00", "stroke-width": 2}));
  if (e.tail) g.appendChild(mk("circle", {cx: e.tail[0], cy: e.tail[1], r: 7, fill: "#2e8b3d"}));
  if (e.head) g.appendChild(mk("circle", {cx: e.head[0], cy: e.head[1], r: 7, fill: "#e0245e"}));
  ov.appendChild(g);
}
for (const n of D.nodes) {
  const g = mk("g", {"data-id": n.id, class: "ovn"});
  g.appendChild(rect(n.box, {stroke: n.color, "stroke-width": 3, "stroke-dasharray": n.conf < 0.5 ? "8 5" : ""}));
  const t = mk("text", {x: n.box[0] + 4, y: n.box[1] + 16, "font-size": 15, "font-weight": 700, fill: n.color});
  t.textContent = n.id; g.appendChild(t); ov.appendChild(g);
}
const tb = (id, rows) => { document.querySelector(`#${id} tbody`).innerHTML = rows.join(""); };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
tb("nodes", D.nodes.map(n => `<tr data-id="${n.id}"><td>${n.id}</td><td>${esc(n.cls)}</td><td>${esc(n.text) || "<i>—</i>"}</td><td>${n.conf.toFixed(2)}${n.conf < 0.5 ? ' <span class="tag warn">low</span>' : ""}</td></tr>`));
tb("edges", D.edges.map(e => `<tr data-id="${e.id}" data-ends="${e.src} ${e.dst}"><td>${e.src} → ${e.dst}<br><span class="meta">${esc(textOf(e.src))} → ${esc(textOf(e.dst))}</span></td><td>${esc(e.label) || "<i>—</i>"}</td><td><span class="tag${e.matched_by !== "iou" ? " warn" : ""}">${e.matched_by === "iou" ? "overlap" : e.matched_by + " (fallback)"}</span></td><td class="meta">${esc(e.endpoints)}</td></tr>`));
const highlight = ids => {
  document.querySelectorAll("[data-id]").forEach(el => {
    const on = ids.includes(el.dataset.id);
    if (el.tagName === "TR") el.classList.toggle("hl", on);
    else el.style.opacity = ids.length && !on ? 0.25 : 1;
  });
};
document.querySelectorAll("tbody tr").forEach(tr => {
  tr.addEventListener("mouseenter", () => highlight([tr.dataset.id, ...(tr.dataset.ends || "").split(" ").filter(Boolean)]));
  tr.addEventListener("mouseleave", () => highlight([]));
});
document.getElementById("copy").onclick = async () => {
  try { await navigator.clipboard.writeText(document.getElementById("src").textContent); document.getElementById("copy").textContent = "Copied"; }
  catch (e) { document.getElementById("copy").textContent = "Select and copy"; }
};
</script>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({startOnLoad: true, theme: "default", flowchart: {curve: "basis"}});
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
