"""Run all GPU training on Kaggle (or any Linux GPU box) with resume and a time budget.

Layout
    OUT  (saved):  /kaggle/working/hdr      -> becomes the notebook's output
        yolov7_runs/<ds>/weights/best.pt   YOLOv7 weights (optimizer stripped)
        deeplab/best.pt, log.jsonl         DeepLabv3 arrow segmentation
        trocr/<exp>/log.jsonl, test_metrics.json, trocr/<best exp>/best/
        logs/*.log                         full console logs
        STATE.json                         what is finished; used to resume
    WORK (scratch): /tmp/hdr_work          datasets, YOLO labels, masks, text crops

Only what later steps need is kept: final weights, logs and metrics. Extra
YOLOv7 epoch checkpoints are disabled, resume checkpoints (last.pt) are
deleted once a part finishes, and only the TrOCR experiment with the best
validation CER keeps its model (the others keep logs and metrics).

Resume: add the previous run's output as an input. `restore` copies any
earlier OUT folder found under /kaggle/input back into place; finished parts
are skipped and unfinished training resumes from last.pt.

Usage (from the notebook):
    python -m hdr.tools.cloud_run prepare
    python -m hdr.tools.cloud_run run --parts A --gpu 0 --deadline <unix time>
    python -m hdr.tools.cloud_run run --parts B,C --gpu 1 --deadline <unix time>
    python -m hdr.tools.cloud_run report
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path(os.environ.get("HDR_CLOUD_OUT", "/kaggle/working/hdr"))
WORK = Path(os.environ.get("HDR_CLOUD_WORK", "/tmp/hdr_work"))
INPUT_ROOT = Path(os.environ.get("HDR_CLOUD_INPUT", "/kaggle/input"))
CODE = Path(__file__).resolve().parents[2]
DATASETS_REPO = "https://github.com/bernhardschaefer/handwritten-diagram-datasets"
YOLOV7_REPO = "https://github.com/WongKinYiu/yolov7"
YOLOV7_WEIGHTS = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt"
YOLO_NC = {"fca": 9, "fcb": 9, "fa": 6}
SAFETY_MIN = 20  # stop launching / kill this many minutes before the deadline

FONTS = [
    "ofl/indieflower/IndieFlower-Regular.ttf", "ofl/shadowsintolight/ShadowsIntoLight.ttf",
    "ofl/reeniebeanie/ReenieBeanie.ttf", "ofl/patrickhand/PatrickHand-Regular.ttf",
    "ofl/kalam/Kalam-Regular.ttf", "ofl/gloriahallelujah/GloriaHallelujah.ttf",
    "ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf",
    "ofl/nanumpenscript/NanumPenScript-Regular.ttf",
    "apache/homemadeapple/HomemadeApple-Regular.ttf",
]
BUNDLED_FONTS = CODE / "hdr" / "assets" / "fonts"

# YOLOv7 train.py writes many extra full checkpoints (init.pt, epoch_000,
# every 25 epochs, the last 5 epochs, every new best after epoch 200).
# On Kaggle's 20 GB output limit that is too much; keep only last.pt + best.pt.
YOLO_SAVE_PATCHES = [
    ("torch.save(model, wdir / 'init.pt')", "pass  # hdr: init.pt disabled"),
    ("if (best_fitness == fi) and (epoch >= 200):", "if False:  # hdr: best_xxx disabled"),
    ("if epoch == 0:", "if False:  # hdr: epoch_000 disabled"),
    ("elif ((epoch+1) % 25) == 0:", "elif False:  # hdr: epoch_xxx disabled"),
    ("elif epoch >= (epochs-5):", "elif False:  # hdr: last-5 checkpoints disabled"),
]


# Kaggle ships wandb; YOLOv7 then calls wandb.init() and crashes asking for an
# API key. Make the import fail so YOLOv7 runs as if wandb were not installed.
YOLO_WANDB_PATCH = [
    ("    import wandb\n    from wandb import init, finish",
     "    raise ImportError('wandb disabled by hdr')  # hdr: no wandb"),
]


def font_ok(path: Path) -> bool:
    from hdr.data.synth_text import font_ok as _ok  # lazy: PIL only needed for part C
    return _ok(path)


class Deadline(Exception):
    """Raised when there is not enough time left to start or finish a step."""


# ---------------------------------------------------------------- helpers

def log(msg: str, log_file: Path | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def minutes_left(deadline: float) -> float:
    return (deadline - time.time()) / 60


def _kill(p: subprocess.Popen) -> None:
    """Kill a command and its children (data-loader workers, git helpers)."""
    if os.name == "posix":
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    else:
        p.kill()


def sh(cmd: str, cwd: Path | str = CODE, deadline: float | None = None,
       env: dict | None = None, log_file: Path | None = None, timeout: float | None = None) -> None:
    """Run a shell command, stream output to stdout and log_file.

    Raises Deadline if the deadline (minus the safety margin) is reached; the
    whole process group is killed so data-loader workers die too. Raises
    RuntimeError on a non-zero exit code, or when `timeout` seconds pass
    (network steps: a hung download must not eat the GPU budget).
    """
    t0 = time.time()
    if deadline is not None and minutes_left(deadline) < SAFETY_MIN:
        raise Deadline(f"not starting, {minutes_left(deadline):.0f} min left: {cmd}")
    log(f"$ {cmd}", log_file)
    full_env = {**os.environ, **(env or {})}
    p = subprocess.Popen(cmd, shell=True, cwd=str(cwd), env=full_env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
                         start_new_session=os.name == "posix")
    out = log_file.open("a", encoding="utf-8") if log_file else None
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in p.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    killed = timed_out = False
    try:
        while True:
            try:
                line = lines.get(timeout=1.0)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                print(line, end="", flush=True)
                if out:
                    out.write(line)
                    out.flush()
            if p.poll() is None:
                if deadline is not None and minutes_left(deadline) < SAFETY_MIN / 2:
                    killed = True
                elif timeout is not None and time.time() - t0 > timeout:
                    timed_out = True
                if killed or timed_out:
                    _kill(p)
                    break
        p.wait()
    finally:
        if out:
            out.close()
    if killed:
        raise Deadline(f"stopped for time budget: {cmd}")
    if timed_out:
        raise RuntimeError(f"timed out after {timeout:.0f}s: {cmd}")
    if p.returncode:
        raise RuntimeError(f"exit code {p.returncode}: {cmd}")


def load_state() -> dict:
    f = OUT / "STATE.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_state(state: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "STATE.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(OUT / "STATE.json")


def set_done(key: str, info: dict | None = None) -> None:
    # Re-read first: two lanes (GPU 0 and GPU 1) write the same file.
    state = load_state()
    state[key] = {"done": True, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}
    save_state(state)


def is_done(key: str) -> bool:
    return load_state().get(key, {}).get("done", False)


REQUIRED_DATASETS = ("fca", "fcb", "fa")


def input_datasets_root() -> Path | None:
    """Folder under /kaggle/input that holds fca/, fcb/ and fa/ (each with train.json)."""
    if not INPUT_ROOT.is_dir():
        return None
    for tj in sorted(INPUT_ROOT.rglob("train.json")):
        root = tj.parent.parent
        if tj.parent.name == "fca" and all((root / d / "train.json").exists() for d in REQUIRED_DATASETS):
            return root
    return None


def datasets_root() -> Path:
    """Uploaded Kaggle dataset if present, else the GitHub copy in WORK."""
    return input_datasets_root() or WORK / "handwritten-diagram-datasets" / "datasets"


def fetch(attempts: list[str], cwd: Path, check: Path, lf: Path | None = None,
          tries: int = 3, timeout: float = 600) -> None:
    """Run download commands with a timeout and retries until `check` exists.

    `attempts` are alternative commands (e.g. git clone, then a zip download);
    each is tried `tries` times. Partial results are removed between tries.
    """
    for cmd in attempts:
        for i in range(1, tries + 1):
            try:
                sh(cmd, cwd=cwd, log_file=lf, timeout=timeout)
                if check.exists():
                    return
                log(f"download finished but {check} is missing", lf)
            except RuntimeError as e:
                log(f"download attempt {i}/{tries} failed: {e}", lf)
            # remove partial results: a half-written file would be skipped by wget -nc
            if check.is_dir():
                shutil.rmtree(check, ignore_errors=True)
            elif check.exists():
                check.unlink()
            time.sleep(10 * i)
    raise RuntimeError(f"could not download {check.name} after all attempts")


def github_fetch(repo_url: str, dest: Path, lf: Path | None = None) -> None:
    """git clone, falling back to GitHub's zip download (both with timeouts and retries)."""
    if dest.is_dir():
        return
    name = repo_url.rstrip("/").split("/")[-1]
    zip_url = repo_url.replace("https://github.com/", "https://codeload.github.com/") + "/zip/refs/heads/main"
    fetch([f"git clone -q --depth 1 {repo_url} {dest.name}",
           f"rm -rf {name}-main {name}.zip && "
           f"wget -q --tries=3 --timeout=60 -O {name}.zip {zip_url} && unzip -q -o {name}.zip "
           f"&& rm -f {name}.zip && mv {name}-main {dest.name}"],
          cwd=dest.parent, check=dest, lf=lf)


def work_env() -> dict:
    return {
        "HDR_DATASETS_ROOT": str(datasets_root()),
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "HDR_YOLO_DATA_ROOT": str(WORK / "data_yolo"),
        "HDR_WORK_ROOT": str(WORK / "data_work"),
        "HDR_OUT_ROOT": str(OUT / "misc"),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",  # YOLOv7 checkpoints are full pickles
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",  # second guard; YOLOv7's import is also patched
    }


def patch_text(path: Path, patches: list[tuple[str, str]]) -> int:
    """Apply exact-string patches once; returns how many were applied now."""
    text = path.read_text(encoding="utf-8")
    applied = 0
    for old, new in patches:
        if old in text:
            text = text.replace(old, new, 1)
            applied += 1
        elif new not in text:
            raise RuntimeError(f"{path.name}: patch target not found: {old!r}")
    path.write_text(text, encoding="utf-8")
    return applied


def find_input_dir(word: str, max_depth: int = 4) -> Path | None:
    """First folder under INPUT_ROOT (any depth up to max_depth) whose name contains `word`."""
    level = [INPUT_ROOT]
    for _ in range(max_depth):
        nxt = []
        for d in level:
            for p in sorted(d.iterdir()) if d.is_dir() else []:
                if p.is_dir():
                    if word in p.name.lower():
                        return p
                    nxt.append(p)
        level = nxt
    return None


# ---------------------------------------------------------------- steps

def restore() -> None:
    """Copy outputs of an earlier run (attached as input) back into OUT."""
    if (OUT / "STATE.json").exists():
        log(f"{OUT} already has STATE.json, nothing to restore")
        return
    # Any depth: Kaggle mounts inputs at /kaggle/input/<name>/ or, newer,
    # /kaggle/input/notebooks/<user>/<name>/.
    candidates = sorted(p for p in INPUT_ROOT.rglob("STATE.json") if p.parent.name == "hdr")
    if not candidates:
        log("no earlier run found under /kaggle/input, starting fresh")
        return
    src = candidates[0].parent
    log(f"restoring earlier run from {src}")
    shutil.copytree(src, OUT, dirs_exist_ok=True)


def prepare(do_restore: bool = True) -> None:
    """Steps shared by all parts: restore outputs, fetch datasets."""
    if do_restore:
        restore()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    found = input_datasets_root()
    if found:
        log(f"datasets from Kaggle input: {found}")
    else:
        log("datasets not in /kaggle/input, downloading from GitHub (timeouts + retries)")
        github_fetch(DATASETS_REPO, WORK / "handwritten-diagram-datasets")


def part_a(datasets: list[str], deadline: float, gpu: int, epochs: int, batch: int, lf: Path) -> None:
    env = {**work_env(), "CUDA_VISIBLE_DEVICES": str(gpu)}
    sh("python -m hdr.data.coco_to_yolo --datasets fca fcb fa", env=env, log_file=lf)
    y = WORK / "yolov7"
    github_fetch(YOLOV7_REPO, y, lf)
    fetch([f"wget -q -nc --tries=3 --timeout=60 {YOLOV7_WEIGHTS}"], cwd=y,
          check=y / "yolov7_training.pt", lf=lf, timeout=900)
    # numpy >= 1.24 / 2.x fixes (checked against yolov7 main), then checkpoint limits
    sh(r'''grep -rl --include=*.py -E "np\.(int|float)\b" . | xargs -r sed -i -E 's/np\.(int|float)\b([^_0-9])/\1\2/g' ''',
       cwd=y, log_file=lf)
    patch_text(y / "utils" / "metrics.py",
               [("np.trapz(", 'getattr(np, "trapezoid", getattr(np, "trapz", None))(')])
    patch_text(y / "train.py", YOLO_SAVE_PATCHES)
    patch_text(y / "utils" / "wandb_logging" / "wandb_utils.py", YOLO_WANDB_PATCH)
    for ds, nc in YOLO_NC.items():
        cfg = (y / "cfg" / "training" / "yolov7.yaml").read_text().replace("nc: 80", f"nc: {nc}", 1)
        (y / "cfg" / "training" / f"yolov7_{ds}.yaml").write_text(cfg)

    runs = OUT / "yolov7_runs"
    for ds in datasets:
        key = f"yolov7_{ds}"
        if is_done(key):
            log(f"[{key}] already done", lf)
            continue
        last = runs / ds / "weights" / "last.pt"
        if not is_done(f"{key}_trained"):
            if last.exists():
                cmd = f"python train.py --resume {last}"
            else:
                cmd = (f"python train.py --workers 2 --device 0 --batch-size {batch} --epochs {epochs} "
                       f"--img 640 640 --data {WORK}/data_yolo/{ds}.yaml --cfg cfg/training/yolov7_{ds}.yaml "
                       f"--weights yolov7_training.pt --hyp data/hyp.scratch.custom.yaml "
                       f"--project {runs} --name {ds} --exist-ok")
            sh(cmd, cwd=y, deadline=deadline, env=env, log_file=lf)
            # Training finished: a later --resume would crash on the finished checkpoint.
            set_done(f"{key}_trained")
        sh(f"python test.py --data {WORK}/data_yolo/{ds}.yaml --weights {runs / ds / 'weights' / 'best.pt'} "
           f"--img 640 --batch 16 --task test --device 0 --name {ds}_test --project {runs} --exist-ok",
           cwd=y, deadline=deadline, env=env, log_file=lf)
        # keep best.pt (optimizer already stripped by YOLOv7); drop the resume checkpoint
        last.unlink(missing_ok=True)
        set_done(key, {"weights": str((runs / ds / "weights" / "best.pt").relative_to(OUT))})


def part_b(deadline: float, gpu: int, lr: float, epochs: int, patience: int, lf: Path,
           limit: int = 0) -> None:
    if is_done("deeplab"):
        log("[deeplab] already done", lf)
        return
    env = {**work_env(), "CUDA_VISIBLE_DEVICES": str(gpu)}
    sh("python -m hdr.data.arrow_masks --datasets fca fcb", env=env, log_file=lf)
    out = OUT / "deeplab"
    sh(f"python -m hdr.train.train_deeplab --out {out} --batch 4 --workers 2 "
       f"--lr {lr} --epochs {epochs} --patience {patience}" + (f" --limit {limit}" if limit else ""),
       deadline=deadline, env=env, log_file=lf)
    (out / "last.pt").unlink(missing_ok=True)
    rows = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines() if l.strip()]
    set_done("deeplab", {"weights": "deeplab/best.pt", "best_val_iou": max(r["val_iou"] for r in rows),
                         "epochs": len(rows), "lr": lr})


def part_c(deadline: float, gpu: int, epochs: int, synth_n: int, lf: Path, limit: int = 0) -> None:
    env = {**work_env(), "CUDA_VISIBLE_DEVICES": str(gpu)}
    text = WORK / "data_work" / "text"
    sh("python -m hdr.data.text_crops", env=env, log_file=lf)
    fonts = WORK / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    # Fonts ship inside the code zip (hdr/assets/fonts). Downloading them from
    # GitHub on Kaggle can return error pages saved as .ttf, so the download
    # below is only a fallback when the bundled fonts are missing.
    for ttf in sorted(BUNDLED_FONTS.glob("*.ttf")):
        if font_ok(ttf):
            shutil.copy2(ttf, fonts / ttf.name)
    bundled = len(list(fonts.glob("*.ttf")))
    log(f"{bundled} bundled handwriting fonts", lf)
    for f in ([] if bundled else FONTS):
        try:
            r = subprocess.run(["wget", "-q", "-nc", "--tries=2", "--timeout=30", "-P", str(fonts),
                                f"https://github.com/google/fonts/raw/main/{f}"], timeout=120)
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        if not ok:
            log(f"font skipped: {f}", lf)
    # A cut-off or retried download can leave a broken .ttf; drop it so the
    # next run downloads it again and synth_text never sees it.
    for ttf in fonts.glob("*.ttf"):
        if not font_ok(ttf):
            log(f"font unreadable, deleted: {ttf.name}", lf)
            ttf.unlink()
    if not any(fonts.glob("*.ttf")):
        raise RuntimeError("no handwriting fonts could be downloaded for synthetic text")
    sh(f"python -m hdr.data.synth_text --n {synth_n} --fonts-dir {fonts}", env=env, log_file=lf)

    crohme_src = find_input_dir("crohme")
    if crohme_src:
        sh(f'python -m hdr.data.crohme --src "{crohme_src}"', env=env, log_file=lf)
    else:
        log("no CROHME input (a /kaggle/input folder with 'crohme' in its name) -> exp2 skipped", lf)

    exps = {"exp1": [text / "fca" / "train.tsv"]}
    if crohme_src:
        exps["exp2"] = exps["exp1"] + [text / "crohme.tsv"]
    exps["exp3"] = exps.get("exp2", exps["exp1"]) + [text / "synth.tsv"]

    out = OUT / "trocr"
    for name, train in exps.items():
        key = f"trocr_{name}"
        if is_done(key):
            log(f"[{key}] already done", lf)
            continue
        shutil.rmtree(out / name, ignore_errors=True)  # an interrupted experiment restarts cleanly
        sh(f"python -m hdr.train.train_trocr --name {name} --train {' '.join(map(str, train))} "
           f"--val {text / 'fca' / 'val.tsv'} --test {text / 'fca' / 'test.tsv'} --out {out} "
           f"--epochs {epochs} --batch 16 --workers 2" + (f" --limit {limit}" if limit else ""),
           deadline=deadline, env=env, log_file=lf)
        m = json.loads((out / name / "test_metrics.json").read_text())
        rows = [json.loads(l) for l in (out / name / "log.jsonl").read_text().splitlines() if l.strip()]
        set_done(key, {"test_wer": m["wer"], "test_cer": m["cer"],
                       "best_val_cer": min(r.get("val_cer", 9e9) for r in rows), "train": [str(t.name) for t in train]})

    # Keep only the model with the best validation CER (chosen on val, not test).
    state = load_state()
    finished = {k[6:]: v for k, v in state.items() if k.startswith("trocr_exp") and v.get("done")}
    if set(finished) >= set(exps):
        best = min(finished, key=lambda n: finished[n]["best_val_cer"])
        for name in finished:
            if name != best:
                shutil.rmtree(out / name / "best", ignore_errors=True)
        set_done("trocr", {"best_exp": best, "weights": f"trocr/{best}/best"})


def report() -> None:
    state = load_state()
    print(json.dumps(state, indent=2))
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9 if OUT.exists() else 0
    print(f"\nOUT size: {total:.2f} GB (Kaggle limit 20 GB)")
    needed = ["yolov7_fca", "yolov7_fcb", "yolov7_fa", "deeplab", "trocr"]
    missing = [k for k in needed if not state.get(k, {}).get("done")]
    print("ALL DONE" if not missing else f"NOT FINISHED: {', '.join(missing)} -> run again with this output attached")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "report"])
    ap.add_argument("--parts", default="A,B,C")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=time.time() + 11 * 3600)
    ap.add_argument("--yolo-datasets", default="fca,fcb,fa")
    ap.add_argument("--yolo-epochs", type=int, default=300)
    ap.add_argument("--yolo-batch", type=int, default=16)
    ap.add_argument("--deeplab-lr", type=float, default=1e-4)
    ap.add_argument("--deeplab-epochs", type=int, default=150)
    ap.add_argument("--deeplab-patience", type=int, default=15)
    ap.add_argument("--trocr-epochs", type=int, default=20)
    ap.add_argument("--synth-lines", type=int, default=5000)
    ap.add_argument("--deeplab-limit", type=int, default=0, help="smoke test: use only N images")
    ap.add_argument("--trocr-limit", type=int, default=0, help="smoke test: use only N lines per tsv")
    ap.add_argument("--no-restore", action="store_true", help="do not copy earlier output (smoke test)")
    args = ap.parse_args()

    if args.cmd == "prepare":
        prepare(do_restore=not args.no_restore)
        return
    if args.cmd == "report":
        report()
        return

    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    lf = OUT / "logs" / f"gpu{args.gpu}_{args.parts.replace(',', '')}.log"
    log(f"lane gpu{args.gpu} parts {args.parts}, {minutes_left(args.deadline):.0f} min budget", lf)
    failed = []
    for part in args.parts.split(","):
        try:
            if part == "A":
                part_a(args.yolo_datasets.split(","), args.deadline, args.gpu,
                       args.yolo_epochs, args.yolo_batch, lf)
            elif part == "B":
                part_b(args.deadline, args.gpu, args.deeplab_lr, args.deeplab_epochs,
                       args.deeplab_patience, lf, limit=args.deeplab_limit)
            elif part == "C":
                part_c(args.deadline, args.gpu, args.trocr_epochs, args.synth_lines, lf,
                       limit=args.trocr_limit)
        except Deadline as e:
            log(f"TIME BUDGET: {e}. Progress is saved; run again with this output attached.", lf)
            break
        except Exception as e:  # keep going: one broken part must not lose the others
            log(f"PART {part} FAILED: {type(e).__name__}: {e}", lf)
            failed.append(part)
    log(f"lane gpu{args.gpu} finished" + (f", failed parts: {failed}" if failed else ""), lf)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
