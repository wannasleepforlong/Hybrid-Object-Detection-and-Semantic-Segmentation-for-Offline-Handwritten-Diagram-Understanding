"""Generate notebooks/train_kaggle.ipynb (all GPU training for the paper, on Kaggle).

The notebook is thin: the work is done by hdr/tools/cloud_run.py, which is
unit-tested locally. Run it with "Save Version -> Save & Run All (Commit)".

Edit the cells here, then run: python -m hdr.tools.build_notebook
"""
from __future__ import annotations

import json

from hdr.config import ROOT

CELLS: list[tuple[str, str]] = [
("md", """# Hybrid diagram recognizer: GPU training on Kaggle

Trains the three models from Costa & Mello (IEEE Access 2026). With **GPU T4 ×2**, two lanes run at the same time:

| Lane | Work | Estimated time |
|---|---|---|
| GPU 0 | A: YOLOv7 on FC_A, FC_B, FA (300 epochs each) | ~7–10 h |
| GPU 1 | B: DeepLabv3 arrows, then C: TrOCR experiments | ~3–5.5 h |
| | **Wall time** (one GPU: run one after the other, ~10–15 h) | **~7–10 h** |

**One-time setup**
1. Kaggle account with phone verification (needed for GPU and Internet).
2. Datasets → New Dataset → upload `hdr_code.zip`, name it **hdr_code** (any name works; the notebook searches all inputs).
3. Datasets → New Dataset → upload `hdr_datasets.zip` (FC_A, FC_B, FA; ~100 MB), name it **hdr_datasets**.
   Without it the run downloads them from GitHub, which can hang or drop on Kaggle.
4. Optional: a dataset whose name contains **crohme** with CROHME `.inkml` files (enables TrOCR exp2).

**Smoke test first (~10–15 min):** keep `SMOKE = True` in Settings and press **Run All** in the editor
(interactive, so you see the output live). Every step runs on tiny settings into `hdr_smoke/`. The last
lines should say **ALL DONE**. Then set `SMOKE = False` for the real run below.

**Run**
1. Import this notebook (File → Import notebook).
2. Add Input → your **hdr_code** and **hdr_datasets** datasets (and the crohme one if you have it).
3. Settings: Accelerator **GPU T4 ×2**, **Internet on**, Persistence: none needed.
4. **Save Version → Save & Run All (Commit)**. It runs in the background; the browser can be closed.

**If it stops before everything is done** (12 h limit, error): open the notebook, Add Input → Notebooks →
*Your Work* → this notebook (the output of the last version), then Save & Run All again.
Finished parts are skipped and training resumes from its last checkpoint.

**Saved output** (`/kaggle/working/hdr`, ~2–3 GB): YOLOv7 `best.pt` ×3, DeepLab `best.pt`, the best TrOCR
model, all logs and metrics, and `STATE.json`. Download: the version's **Output** tab, or on your PC
`kaggle kernels output <user>/<notebook> -p weights/`."""),

("md", "## Settings"),
("code", """import os, time
START = time.time()

# SMOKE = True: ~10-15 min check that every step runs (tiny settings, separate output folder
# /kaggle/working/hdr_smoke, never resumes). Set SMOKE = False for the real overnight run.
SMOKE = True

BUDGET_HOURS = 11.0        # Kaggle stops a run at 12 h; leave time to save the output
YOLO_DATASETS = 'fca,fcb,fa'
YOLO_EPOCHS = 300          # paper: 300
YOLO_BATCH = 16
DEEPLAB_LR = 1e-4          # paper: 1e-5 with 10,000 epochs; too slow for this budget (see PLAN.md)
DEEPLAB_EPOCHS = 150       # upper bound, early stopping usually ends sooner
DEEPLAB_PATIENCE = 15
TROCR_EPOCHS = 20
SYNTH_LINES = 5000
DEEPLAB_LIMIT = TROCR_LIMIT = 0   # 0 = use all data

if SMOKE:
    YOLO_EPOCHS, DEEPLAB_EPOCHS, TROCR_EPOCHS, SYNTH_LINES = 2, 2, 1, 200
    DEEPLAB_LIMIT, TROCR_LIMIT = 40, 64
OUT_DIR = '/kaggle/working/hdr_smoke' if SMOKE else '/kaggle/working/hdr'
os.environ['HDR_CLOUD_OUT'] = OUT_DIR
print('SMOKE TEST' if SMOKE else 'FULL RUN', '->', OUT_DIR)"""),

("md", "## Setup"),
("code", """import glob, os, shutil, subprocess, sys, torch
n_gpu = torch.cuda.device_count()
assert n_gpu > 0, 'No GPU: Settings -> Accelerator -> GPU T4 x2'
print(n_gpu, 'GPU(s):', [torch.cuda.get_device_name(i) for i in range(n_gpu)])

found = sorted(glob.glob('/kaggle/input/**/hdr/config.py', recursive=True))
assert found, 'Code not found: Add Input -> your "hdr_code" dataset'
def build_of(cfg):  # stamp written by hdr.tools.pack_code; old zips have none
    b = os.path.join(os.path.dirname(cfg), 'BUILD.txt')
    return open(b).read().strip() if os.path.exists(b) else '0000 (no build stamp)'
for cfg in found:
    print('code found:', build_of(cfg), os.path.dirname(os.path.dirname(cfg)))
newest = max(found, key=build_of)   # several attached (old + new version): use the newest
CODE = '/tmp/hdr_code'
shutil.rmtree(CODE, ignore_errors=True)
shutil.copytree(os.path.dirname(os.path.dirname(newest)), CODE)
print('USING code build', build_of(newest), 'from', os.path.dirname(os.path.dirname(newest)))

def stream(cmd, cwd=CODE):
    \"\"\"Run a command, print its output live, stop the notebook if it fails.\"\"\"
    print('$', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait():
        raise RuntimeError(f'failed with exit code {p.returncode}')

try:
    stream([sys.executable, '-m', 'pip', '-q', 'install', 'jiwer', 'pytest'])
except RuntimeError:
    raise RuntimeError('pip failed: turn on Settings -> Internet (needs a phone-verified account)')
stream([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'])
DEADLINE = START + BUDGET_HOURS * 3600
COMMON = ['--deadline', str(DEADLINE), '--yolo-datasets', YOLO_DATASETS, '--yolo-epochs', str(YOLO_EPOCHS),
          '--yolo-batch', str(YOLO_BATCH), '--deeplab-lr', str(DEEPLAB_LR), '--deeplab-epochs', str(DEEPLAB_EPOCHS),
          '--deeplab-patience', str(DEEPLAB_PATIENCE), '--trocr-epochs', str(TROCR_EPOCHS),
          '--synth-lines', str(SYNTH_LINES), '--deeplab-limit', str(DEEPLAB_LIMIT),
          '--trocr-limit', str(TROCR_LIMIT)]
RUN = [sys.executable, '-m', 'hdr.tools.cloud_run']"""),

("md", "## Restore earlier output, fetch datasets"),
("code", """stream(RUN + ['prepare'] + (['--no-restore'] if SMOKE else []))"""),

("md", """## Train

With 2 GPUs, B+C run in the background on GPU 1 (log: `hdr/logs/gpu1_BC.log`) while A runs here on GPU 0.
With 1 GPU everything runs here, one after the other."""),
("code", """# Errors are reported but do not stop the notebook, so the output (checkpoints, logs) is always saved.
def lane(cmd):
    try:
        stream(cmd)
    except RuntimeError as e:
        print('LANE ERROR:', e, '- see hdr/logs/ and the report below', flush=True)

if n_gpu >= 2:
    lane1 = subprocess.Popen(RUN + ['run', '--parts', 'B,C', '--gpu', '1'] + COMMON, cwd=CODE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    lane(RUN + ['run', '--parts', 'A', '--gpu', '0'] + COMMON)
    print('GPU 0 lane finished, waiting for GPU 1 lane ...', flush=True)
    lane1.wait()
    print('GPU 1 lane exit code', lane1.returncode)
    log1 = f'{OUT_DIR}/logs/gpu1_BC.log'
    if os.path.exists(log1):
        print(''.join(open(log1).readlines()[-40:]))
else:
    lane(RUN + ['run', '--parts', 'B,C,A', '--gpu', '0'] + COMMON)"""),

("md", "## Report"),
("code", """stream(RUN + ['report'])
for f in sorted(glob.glob(f'{OUT_DIR}/**/*', recursive=True)):
    if os.path.isfile(f) and os.path.getsize(f) > 5e6:
        print(f'{os.path.getsize(f) / 1e6:8.0f} MB  {f}')"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": lines})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                       "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    dst = ROOT / "notebooks" / "train_kaggle.ipynb"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    old = ROOT / "notebooks" / "train_colab.ipynb"
    if old.exists():
        old.unlink()
    print("wrote", dst)


if __name__ == "__main__":
    main()
