import random
from pathlib import Path

import numpy as np
import pytest

from hdr.data.arrow_masks import ink_mask, polyline_mask
from hdr.data.coco import Annotation, ImageRecord
from hdr.data.crohme import latex_to_plain
from hdr.data.synth_text import PATTERNS, make_text


@pytest.mark.parametrize("latex,plain", [
    ("$a+b=c$", "a+b=c"),
    ("$x \\times y$", "x*y"),
    ("$a \\div 2$", "a/2"),
    ("$\\frac{a}{b}$", None),
    ("$x^{2}$", None),
    ("$\\sqrt{x}$", None),
    ("$\\alpha + 1$", None),
])
def test_latex_to_plain(latex, plain):
    assert latex_to_plain(latex) == plain


WIN_FONT = Path("C:/Windows/Fonts/Inkfree.ttf")


def test_font_ok_rejects_broken_file(tmp_path):
    from hdr.data.synth_text import font_ok
    broken = tmp_path / "cut.ttf"
    broken.write_bytes(b"<html>not a font</html>")
    assert not font_ok(broken)


@pytest.mark.skipif(not WIN_FONT.exists(), reason="needs a real .ttf (Windows font)")
def test_find_fonts_skips_broken_and_keeps_good(tmp_path):
    import shutil
    from hdr.data.synth_text import find_fonts
    shutil.copy(WIN_FONT, tmp_path / "good.ttf")
    (tmp_path / "cut.ttf").write_bytes(b"\x00\x01partial")
    assert [f.name for f in find_fonts(tmp_path)] == ["good.ttf"]


def test_part_c_deletes_broken_font(tmp_path, monkeypatch):
    from hdr.tools import cloud_run as cr
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    (fonts / "cut.ttf").write_bytes(b"partial")
    monkeypatch.setattr(cr, "WORK", tmp_path)
    monkeypatch.setattr(cr, "OUT", tmp_path / "out")
    monkeypatch.setattr(cr, "BUNDLED_FONTS", tmp_path / "no_bundled_fonts")   # force the download path
    monkeypatch.setattr(cr, "sh", lambda *a, **k: None)
    monkeypatch.setattr(cr.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    with pytest.raises(RuntimeError, match="no handwriting fonts"):
        cr.part_c(0, 0, 1, 10, None)
    assert not (fonts / "cut.ttf").exists()          # deleted, so the next run downloads it again


def test_bundled_fonts_are_valid_and_used_without_download(tmp_path, monkeypatch):
    from hdr.data.synth_text import BUNDLED_FONTS, font_ok
    from hdr.tools import cloud_run as cr
    bundled = sorted(BUNDLED_FONTS.glob("*.ttf"))
    assert len(bundled) >= 5 and all(font_ok(f) for f in bundled)
    monkeypatch.setattr(cr, "WORK", tmp_path)
    monkeypatch.setattr(cr, "OUT", tmp_path / "out")
    calls = []
    monkeypatch.setattr(cr, "sh", lambda cmd, **k: calls.append(cmd))
    def no_download(*a, **k):
        raise AssertionError("fonts must not be downloaded when bundled fonts exist")
    monkeypatch.setattr(cr.subprocess, "run", no_download)
    try:
        cr.part_c(0, 0, 1, 10, None)
    except FileNotFoundError:
        pass                                          # stops later at the faked TrOCR step; fonts came first
    assert len(list((tmp_path / "fonts").glob("*.ttf"))) == len(bundled)
    assert any("synth_text" in c for c in calls)


def test_synth_patterns_cover_paper_cases():
    rng = random.Random(0)
    texts = [p(rng) for p in PATTERNS for _ in range(50)]
    assert any(t.lower().startswith(("read", "input")) for t in texts)
    assert any("*" in t for t in texts)
    assert all("=" in t for t in texts if not t.lower().startswith(("read", "input")))
    assert make_text(rng)


def _arrow_image():
    """White 100x100 image, vertical arrow shaft x=50 from y=10..80, a node box below."""
    gray = np.full((100, 100), 255, np.uint8)
    gray[10:80, 49:52] = 0                  # shaft
    gray[80:95, 20:80] = 0                  # filled node (ink inside its box)
    arrow = Annotation(id=1, aid=None, category="arrow", bbox=(45, 10, 10, 70),
                       keypoints=[50, 10, 2, 50, 80, 2], waypoints=[[50, 10], [50, 80]])
    node = Annotation(id=2, aid=None, category="process", bbox=(20, 80, 60, 15))
    im = ImageRecord(id=0, file_name="x.png", path=Path("x.png"), width=100, height=100,
                     annotations=[arrow, node])
    return im, gray


def test_ink_mask_keeps_shaft_and_drops_node_interior():
    im, gray = _arrow_image()
    m = ink_mask(im, gray, band=9)
    assert m[40, 50] == 255                 # shaft ink kept
    assert m[40, 60] == 0                   # background not marked
    assert m[88, 50] == 0                   # ink inside node box removed


@pytest.mark.parametrize("category", ["decision", "terminator", "connection"])
def test_node_region_keeps_bbox_corners_free(category):
    from hdr.data.arrow_masks import node_region
    inkmap = np.zeros((200, 200), np.uint8)
    node = Annotation(id=1, aid=None, category=category, bbox=(50, 50, 100, 60))
    m = node_region(node, inkmap, grow=3)
    assert m[80, 100] == 255                # centre removed
    assert m[52, 52] == 0                   # bbox corner kept (arrowheads can end there)


def test_node_region_process_is_exact_box():
    from hdr.data.arrow_masks import node_region
    node = Annotation(id=1, aid=None, category="process", bbox=(50, 50, 100, 60))
    m = node_region(node, np.zeros((200, 200), np.uint8))
    assert m[50, 50] == 255 and m[49, 50] == 0


def test_polyline_mask_has_requested_width():
    im, _ = _arrow_image()
    m = polyline_mask(im, width=6)
    assert 5 <= (m[40] > 0).sum() <= 7
