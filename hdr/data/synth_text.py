"""Render synthetic flowchart text lines (paper section III-D, Table 1).

Four patterns, as described in the paper:
  1. read up to two variables        "read a", "Input a, b"
  2. assignment from variables       "x = y", "s = s + i"
  3. assignment of discrete values   "x = 5", "i = 0"
  4. multiplication of variables     "p = a * b"

Text is drawn with handwriting-style TTF fonts, then distorted (rotation,
shear, stroke thickness, blur, noise) to look closer to pen strokes.

Output: data_work/text/synth/<name>.png and data_work/text/synth.tsv
Usage: python -m hdr.data.synth_text --n 5000 --fonts-dir fonts/
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hdr import config

READ_WORDS = ["read", "Read", "input", "Input", "READ", "INPUT"]
SINGLE_VARS = list("abcdefghiknmpqrstxyzABCDEFMNPST")
WORD_VARS = ["sum", "max", "min", "cnt", "res", "tmp", "fat", "avg"]


def _var(rng: random.Random) -> str:
    if rng.random() < 0.8:
        return rng.choice(SINGLE_VARS)
    return rng.choice(WORD_VARS)


def _eq(rng: random.Random) -> str:
    return rng.choice(["=", " = ", "= ", " ="])


def pattern_read(rng: random.Random) -> str:
    vs = [_var(rng) for _ in range(rng.choice([1, 2]))]
    sep = rng.choice([",", ", "])
    return f"{rng.choice(READ_WORDS)} {sep.join(vs)}"


def pattern_assign_var(rng: random.Random) -> str:
    x = _var(rng)
    form = rng.randrange(3)
    if form == 0:
        rhs = _var(rng)
    elif form == 1:
        rhs = f"{x}{rng.choice(['+', '-', ' + ', ' - '])}{rng.choice([_var(rng), str(rng.randint(1, 9))])}"
    else:
        rhs = f"{_var(rng)}{rng.choice(['+', '-', '/'])}{_var(rng)}"
    return f"{x}{_eq(rng)}{rhs}"


def pattern_assign_value(rng: random.Random) -> str:
    val = str(rng.choice([0, 1, 2, rng.randint(0, 100)]))
    return f"{_var(rng)}{_eq(rng)}{val}"


def pattern_multiply(rng: random.Random) -> str:
    return f"{_var(rng)}{_eq(rng)}{_var(rng)}{rng.choice(['*', ' * '])}{_var(rng)}"


PATTERNS = [pattern_read, pattern_assign_var, pattern_assign_value, pattern_multiply]


def make_text(rng: random.Random) -> str:
    return rng.choice(PATTERNS)(rng)


def render(text: str, font_path: Path, rng: random.Random) -> np.ndarray:
    size = rng.randint(36, 64)
    font = ImageFont.truetype(str(font_path), size)
    left, top, right, bottom = font.getbbox(text)
    pad = size // 2
    img = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=0)
    arr = np.array(img)

    # stroke thickness: erode = thicker dark strokes, dilate = thinner
    k = rng.choice([0, 0, 1, 2])
    if k:
        arr = cv2.erode(arr, np.ones((k + 1, k + 1), np.uint8))
    elif size >= 50 and rng.random() < 0.2:
        arr = cv2.dilate(arr, np.ones((2, 2), np.uint8))

    # rotation + shear
    h, w = arr.shape
    angle = rng.uniform(-4, 4)
    shear = rng.uniform(-0.25, 0.25)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[0, 1] += shear
    m[0, 2] -= shear * h / 2
    arr = cv2.warpAffine(arr, m, (w, h), borderValue=255)

    if rng.random() < 0.5:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
    if rng.random() < 0.3:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 8, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # trim to ink with a small margin, like the real crops
    ys, xs = np.where(arr < 128)
    if len(xs):
        m_ = 4
        arr = arr[max(0, ys.min() - m_):ys.max() + m_ + 1, max(0, xs.min() - m_):xs.max() + m_ + 1]
    return arr


def font_ok(path: Path) -> bool:
    """True if PIL can load the file as a font (a cut-off download cannot)."""
    try:
        ImageFont.truetype(str(path), 24).getbbox("x = a * b")
        return True
    except Exception:
        return False


BUNDLED_FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"


def find_fonts(fonts_dir: Path | None) -> list[Path]:
    fonts = sorted(fonts_dir.glob("*.ttf")) if fonts_dir else []
    if not fonts:
        fonts = sorted(BUNDLED_FONTS.glob("*.ttf"))
    if not fonts:
        win = Path("C:/Windows/Fonts")
        fonts = [win / f for f in ("Inkfree.ttf", "segoepr.ttf", "segoesc.ttf", "comic.ttf")
                 if (win / f).exists()]
    bad = [f for f in fonts if not font_ok(f)]
    for f in bad:
        print(f"skipping unreadable font: {f.name}")
    fonts = [f for f in fonts if f not in bad]
    if not fonts:
        raise SystemExit("No usable handwriting fonts found. Pass --fonts-dir with .ttf files.")
    return fonts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--fonts-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=config.TEXT_ROOT / "synth")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fonts = find_fonts(args.fonts_dir)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        text = make_text(rng)
        arr = render(text, rng.choice(fonts), rng)
        name = f"synth_{i:06d}.png"
        cv2.imwrite(str(args.out / name), arr)
        rows.append(f"{args.out.name}/{name}\t{text}")
    (args.out.parent / f"{args.out.name}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{args.n} synthetic lines with {len(fonts)} fonts -> {args.out}")


if __name__ == "__main__":
    main()
