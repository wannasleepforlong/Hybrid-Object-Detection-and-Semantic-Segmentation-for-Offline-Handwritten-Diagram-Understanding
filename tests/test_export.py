import xml.etree.ElementTree as ET

from hdr.export import layout_direction, route, to_drawio, to_mermaid, to_svg
from hdr.graph import Edge, Graph, Node


def _graph():
    nodes = [Node("n0", "terminator", (100, 0, 200, 40), "Start", 0.95),
             Node("n1", "decision", (100, 100, 200, 160), "x > 0", 0.9),
             Node("n2", "process", (0, 220, 80, 260), "x = 1", 0.3),         # low confidence
             Node("n3", "data", (100, 300, 200, 340), "print x", 0.9)]
    edges = [Edge("n0", "n1", tail=(150, 40), head=(150, 100)),
             Edge("n1", "n2", "No", tail=(100, 130), head=(40, 220)),        # needs an L
             Edge("n1", "n3", "Yes", tail=(150, 160), head=(150, 300), matched_by="nearest")]
    return Graph(nodes, edges, dangling=[5], size=(300, 400))


def test_mermaid_uses_more_of_the_output():
    m = to_mermaid(_graph())
    assert m.startswith("flowchart TD")                       # arrows mostly vertical
    assert 'n1 -.->|"Yes"| n3' in m                            # fallback link drawn dashed
    assert 'n1 -->|"No"| n2' in m
    assert "classDef decision" in m and "class n1 decision" in m
    assert "class n2 lowconf" in m                             # low-confidence symbol marked
    assert "1 arrow(s) had an unmatched end" in m


def test_layout_direction_follows_drawing():
    g = Graph([], [Edge("a", "b", tail=(0, 0), head=(300, 10))])
    assert layout_direction(g) == "LR"


def test_drawio_keeps_positions_and_attachment_points():
    root = ET.fromstring(to_drawio(_graph()))                 # well-formed XML
    cells = {c.get("id"): c for c in root.iter("mxCell")}
    geo = cells["n1"].find("mxGeometry")
    assert (geo.get("x"), geo.get("y"), geo.get("width"), geo.get("height")) == ("100", "100", "100", "60")
    assert "rhombus" in cells["n1"].get("style")
    e0 = cells["e0"].get("style")
    assert "exitX=0.50;exitY=1.00" in e0 and "entryX=0.50;entryY=0.00" in e0   # bottom -> top
    assert "dashed=1" in cells["e2"].get("style")
    assert cells["e1"].get("value") == "No"


def test_route_makes_l_shape_around_symbols():
    g = _graph()
    by_id = {n.id: n for n in g.nodes}
    assert route(g.edges[0], by_id) == [(150, 40), (150, 100)]          # straight
    pts = route(g.edges[1], by_id)
    assert len(pts) == 3 and pts[1] in [(100, 220), (40, 130)]


def test_svg_is_valid_and_complete():
    root = ET.fromstring(to_svg(_graph()))
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f".//{ns}polyline")) == 3
    texts = [t.text for t in root.iter(f"{ns}text")]
    assert "Start" in texts and "Yes" in texts


def test_graph_json_roundtrip():
    from hdr.graph import load_graph
    g = _graph()
    g2 = load_graph(g.to_json())
    assert g2.nodes == g.nodes and g2.edges == g.edges and g2.size == g.size and g2.dangling == g.dangling
