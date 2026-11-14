"""Zip the project for upload to Kaggle.

    python -m hdr.tools.pack_code             ->  out/hdr_code.zip      (code, ~100 KB)
    python -m hdr.tools.pack_code --datasets  ->  out/hdr_datasets.zip  (FC_A, FC_B, FA, ~100 MB)

Upload each as its own Kaggle dataset and add both as notebook inputs. The
runner finds the datasets under /kaggle/input and only falls back to
downloading them from GitHub when they are missing.
"""
from __future__ import annotations

import argparse
import time
import zipfile

from hdr.config import DATASETS_ROOT, OUT_ROOT, ROOT

INCLUDE = ["hdr", "tests", "notebooks", "requirements.txt", "PLAN.md"]
DATASETS = ["fca", "fcb", "fa"]  # the three the paper uses; fcb_scan and hdBPMN are not needed


def pack_code() -> None:
    dst = OUT_ROOT / "hdr_code.zip"
    n = 0
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        # Build stamp: the notebook picks the newest code when several
        # versions/datasets are attached, and prints which one it runs.
        z.writestr("hdr/BUILD.txt", stamp + "\n")
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc":
                    continue
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"{n} files, build {stamp} -> {dst}")


def pack_datasets() -> None:
    dst = OUT_ROOT / "hdr_datasets.zip"
    n = 0
    # PNGs are already compressed; store them to save time.
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for ds in DATASETS:
            for f in sorted((DATASETS_ROOT / ds).rglob("*")):
                if f.is_file():
                    z.write(f, f"datasets/{f.relative_to(DATASETS_ROOT).as_posix()}")
                    n += 1
    print(f"{n} files ({dst.stat().st_size / 1e6:.0f} MB) -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", action="store_true", help="also zip FC_A, FC_B and FA")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pack_code()
    if args.datasets:
        pack_datasets()


if __name__ == "__main__":
    main()
