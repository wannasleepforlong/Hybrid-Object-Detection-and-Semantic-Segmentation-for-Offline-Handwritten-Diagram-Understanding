import json
import sys
import time
from pathlib import Path

import pytest

from hdr.tools import cloud_run as cr


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    out, inp = tmp_path / "working" / "hdr", tmp_path / "input"
    inp.mkdir(parents=True)
    monkeypatch.setattr(cr, "OUT", out)
    monkeypatch.setattr(cr, "INPUT_ROOT", inp)
    return out, inp


def test_sh_streams_to_log(tmp_path):
    lf = tmp_path / "x.log"
    cr.sh(f'"{sys.executable}" -c "print(\'hello\')"', cwd=tmp_path, log_file=lf)
    assert "hello" in lf.read_text()


def test_sh_raises_on_failure(tmp_path):
    with pytest.raises(RuntimeError):
        cr.sh(f'"{sys.executable}" -c "raise SystemExit(3)"', cwd=tmp_path)


def test_sh_refuses_to_start_close_to_deadline(tmp_path):
    with pytest.raises(cr.Deadline):
        cr.sh("echo hi", cwd=tmp_path, deadline=time.time() + 60)  # 1 min < SAFETY_MIN


def test_sh_kills_at_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "SAFETY_MIN", 0.1)          # 6 s margin, kill at 3 s left
    t0 = time.time()
    with pytest.raises(cr.Deadline):
        cr.sh(f'"{sys.executable}" -c "import time; [(print(i, flush=True), time.sleep(0.5)) for i in range(60)]"',
              cwd=tmp_path, deadline=t0 + 10)
    assert time.time() - t0 < 20


def test_sh_timeout_kills_hung_command(tmp_path):
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        cr.sh(f'"{sys.executable}" -c "import time; time.sleep(60)"', cwd=tmp_path, timeout=3)
    assert time.time() - t0 < 15


def test_fetch_retries_then_uses_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(cr.time, "sleep", lambda s: None)
    dest = tmp_path / "repo"
    py = f'"{sys.executable}" -c'
    fail = f'{py} "raise SystemExit(128)"'                       # e.g. git clone dropping
    ok = f'{py} "import os; os.makedirs(\'repo\')"'               # fallback download works
    cr.fetch([fail, ok], cwd=tmp_path, check=dest, tries=2, timeout=30)
    assert dest.is_dir()


def test_fetch_gives_up_after_all_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(cr.time, "sleep", lambda s: None)
    fail = f'"{sys.executable}" -c "raise SystemExit(1)"'
    with pytest.raises(RuntimeError, match="could not download"):
        cr.fetch([fail], cwd=tmp_path, check=tmp_path / "x", tries=2, timeout=30)


def test_datasets_found_in_kaggle_input(dirs):
    _, inp = dirs
    root = inp / "datasets" / "someuser" / "hdr-datasets" / "datasets"
    for d in ("fca", "fcb", "fa"):
        (root / d).mkdir(parents=True)
        (root / d / "train.json").write_text("{}")
    assert cr.datasets_root() == root
    assert cr.work_env()["HDR_DATASETS_ROOT"] == str(root)


def test_datasets_fall_back_to_github_copy_when_incomplete(dirs):
    _, inp = dirs
    (inp / "x" / "fca").mkdir(parents=True)
    (inp / "x" / "fca" / "train.json").write_text("{}")         # fcb and fa missing
    assert cr.datasets_root() == cr.WORK / "handwritten-diagram-datasets" / "datasets"


def test_patch_text_is_idempotent_and_strict(tmp_path):
    f = tmp_path / "train.py"
    f.write_text("if epoch == 0:\n    save()\n")
    patches = [("if epoch == 0:", "if False:  # off")]
    assert cr.patch_text(f, patches) == 1
    assert cr.patch_text(f, patches) == 0
    assert "if False:" in f.read_text()
    with pytest.raises(RuntimeError):
        cr.patch_text(f, [("missing line", "x")])


def test_state_roundtrip(dirs):
    assert not cr.is_done("deeplab")
    cr.set_done("deeplab", {"best_val_iou": 0.8})
    cr.set_done("yolov7_fca")
    s = json.loads((dirs[0] / "STATE.json").read_text())
    assert s["deeplab"]["best_val_iou"] == 0.8 and s["yolov7_fca"]["done"]


def test_restore_copies_previous_output(dirs):
    out, inp = dirs
    prev = inp / "my-notebook" / "hdr"
    (prev / "deeplab").mkdir(parents=True)
    (prev / "deeplab" / "last.pt").write_text("ckpt")
    (prev / "STATE.json").write_text(json.dumps({"yolov7_fca": {"done": True}}))
    cr.restore()
    assert (out / "deeplab" / "last.pt").read_text() == "ckpt"
    assert cr.is_done("yolov7_fca")


def fake_wget(cmd, **kw):
    """Stands in for the font download: writes a .ttf into the -P folder."""
    folder = Path(cmd[cmd.index("-P") + 1])
    (folder / cmd[-1].rsplit("/", 1)[-1]).write_text("font")
    return type("R", (), {"returncode": 0})()


class FakeSh:
    """Stands in for cr.sh: records commands and creates the files training would."""

    def __init__(self, out, val_cer=None):
        self.out, self.cmds, self.val_cer = out, [], val_cer or {}

    def __call__(self, cmd, cwd=None, deadline=None, env=None, log_file=None, timeout=None):
        self.cmds.append(cmd)
        if "yolov7_training.pt" in cmd:
            (Path(cwd) / "yolov7_training.pt").write_text("weights")
        if "train.py" in cmd and "--resume" not in cmd or "train.py --resume" in cmd:
            ds = cmd.split("--name ")[1].split()[0] if "--name" in cmd else cmd.split("/")[-3]
            w = self.out / "yolov7_runs" / ds / "weights"
            w.mkdir(parents=True, exist_ok=True)
            (w / "best.pt").write_text("best")
            (w / "last.pt").write_text("last")
        if "train_deeplab" in cmd:
            d = self.out / "deeplab"
            d.mkdir(parents=True, exist_ok=True)
            for f in ("best.pt", "last.pt"):
                (d / f).write_text(f)
            (d / "log.jsonl").write_text('{"val_iou": 0.5}\n{"val_iou": 0.7}\n')
        if "train_trocr" in cmd:
            name = cmd.split("--name ")[1].split()[0]
            d = self.out / "trocr" / name
            (d / "best").mkdir(parents=True, exist_ok=True)
            (d / "log.jsonl").write_text(json.dumps({"val_cer": self.val_cer[name]}) + "\n")
            (d / "test_metrics.json").write_text(json.dumps({"wer": 0.5, "cer": 0.2}))


def _fake_yolov7(work):
    y = work / "yolov7"
    (y / "utils").mkdir(parents=True)
    (y / "cfg" / "training").mkdir(parents=True)
    (y / "train.py").write_text("\n".join(old for old, _ in cr.YOLO_SAVE_PATCHES) + "\n")
    (y / "utils" / "metrics.py").write_text("ap = np.trapz(x)\n")
    (y / "utils" / "wandb_logging").mkdir()
    (y / "utils" / "wandb_logging" / "wandb_utils.py").write_text(
        "try:\n    import wandb\n    from wandb import init, finish\nexcept ImportError:\n    wandb = None\n")
    (y / "cfg" / "training" / "yolov7.yaml").write_text("nc: 80\n")


def test_part_a_trains_tests_and_cleans(dirs, tmp_path, monkeypatch):
    out, _ = dirs
    monkeypatch.setattr(cr, "WORK", tmp_path / "work")
    _fake_yolov7(tmp_path / "work")
    fake = FakeSh(out)
    monkeypatch.setattr(cr, "sh", fake)
    cr.part_a(["fca"], time.time() + 9999, 0, 300, 16, None)
    w = out / "yolov7_runs" / "fca" / "weights"
    assert (w / "best.pt").exists() and not (w / "last.pt").exists()
    assert cr.is_done("yolov7_fca")
    assert "hdr: epoch_000 disabled" in (tmp_path / "work" / "yolov7" / "train.py").read_text()
    ns = {}
    exec((tmp_path / "work" / "yolov7" / "utils" / "wandb_logging" / "wandb_utils.py").read_text(), ns)
    assert ns["wandb"] is None                                   # YOLOv7 sees no wandb
    assert "nc: 9" in (tmp_path / "work" / "yolov7" / "cfg" / "training" / "yolov7_fca.yaml").read_text()
    n = len(fake.cmds)
    cr.part_a(["fca"], time.time() + 9999, 0, 300, 16, None)   # second run: nothing to train
    assert not any("train.py" in c for c in fake.cmds[n:])


def test_part_a_after_training_only_tests(dirs, tmp_path, monkeypatch):
    """Training finished but test.py was cut off: must not --resume a finished run."""
    out, _ = dirs
    monkeypatch.setattr(cr, "WORK", tmp_path / "work")
    _fake_yolov7(tmp_path / "work")
    cr.set_done("yolov7_fca_trained")
    fake = FakeSh(out)
    monkeypatch.setattr(cr, "sh", fake)
    cr.part_a(["fca"], time.time() + 9999, 0, 300, 16, None)
    assert not any("train.py" in c for c in fake.cmds)
    assert any("test.py" in c for c in fake.cmds)
    assert cr.is_done("yolov7_fca")


def test_part_b_keeps_best_and_drops_resume_checkpoint(dirs, monkeypatch):
    out, _ = dirs
    fake = FakeSh(out)
    monkeypatch.setattr(cr, "sh", fake)
    cr.part_b(time.time() + 9999, 1, 1e-4, 150, 15, None)
    assert (out / "deeplab" / "best.pt").exists() and not (out / "deeplab" / "last.pt").exists()
    assert cr.load_state()["deeplab"]["best_val_iou"] == 0.7
    n = len(fake.cmds)
    cr.part_b(time.time() + 9999, 1, 1e-4, 150, 15, None)
    assert len(fake.cmds) == n                                  # skipped when done


def test_part_c_keeps_only_best_validation_model(dirs, monkeypatch):
    out, _ = dirs
    fake = FakeSh(out, val_cer={"exp1": 0.30, "exp3": 0.20})
    monkeypatch.setattr(cr, "sh", fake)
    monkeypatch.setattr(cr.subprocess, "run", fake_wget)
    monkeypatch.setattr(cr, "font_ok", lambda path: True)  # fake fonts are text files
    cr.part_c(time.time() + 9999, 1, 20, 100, None)
    s = cr.load_state()
    assert s["trocr"]["best_exp"] == "exp3"
    assert (out / "trocr" / "exp3" / "best").exists()
    assert not (out / "trocr" / "exp1" / "best").exists()
    assert (out / "trocr" / "exp1" / "test_metrics.json").exists()   # metrics kept for the paper table


def test_smoke_limits_reach_training_commands(dirs, monkeypatch):
    out, _ = dirs
    fake = FakeSh(out, val_cer={"exp1": 0.3, "exp3": 0.2})
    monkeypatch.setattr(cr, "sh", fake)
    monkeypatch.setattr(cr.subprocess, "run", fake_wget)
    monkeypatch.setattr(cr, "font_ok", lambda path: True)  # fake fonts are text files
    cr.part_b(time.time() + 9999, 1, 1e-4, 2, 15, None, limit=40)
    cr.part_c(time.time() + 9999, 1, 1, 200, None, limit=64)
    assert any("train_deeplab" in c and "--limit 40" in c for c in fake.cmds)
    assert all("--limit 64" in c for c in fake.cmds if "train_trocr" in c)
    assert any("synth_text --n 200" in c for c in fake.cmds)


def test_prepare_without_restore_ignores_earlier_output(dirs, tmp_path, monkeypatch):
    out, inp = dirs
    prev = inp / "nb" / "hdr"
    prev.mkdir(parents=True)
    (prev / "STATE.json").write_text(json.dumps({"deeplab": {"done": True}}))
    monkeypatch.setattr(cr, "WORK", tmp_path / "work")
    (tmp_path / "work" / "handwritten-diagram-datasets").mkdir(parents=True)  # skip clone
    cr.prepare(do_restore=False)
    assert not cr.is_done("deeplab")


def test_restore_finds_new_kaggle_layout(dirs):
    out, inp = dirs
    prev = inp / "notebooks" / "someuser" / "hdr-train" / "hdr"
    prev.mkdir(parents=True)
    (prev / "STATE.json").write_text(json.dumps({"deeplab": {"done": True}}))
    cr.restore()
    assert cr.is_done("deeplab")


def test_find_input_dir_any_depth(dirs):
    _, inp = dirs
    (inp / "datasets" / "someuser" / "my-CROHME-2019").mkdir(parents=True)
    assert cr.find_input_dir("crohme").name == "my-CROHME-2019"
    assert cr.find_input_dir("nothing") is None


def test_restore_ignores_code_dataset(dirs):
    out, inp = dirs
    (inp / "hdr-code" / "hdr").mkdir(parents=True)     # code upload: no STATE.json
    cr.restore()
    assert not out.exists()
