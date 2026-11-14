"""Fine-tune TrOCR on flowchart text (paper sections III-D and IV-D).

Training data is one or more TSV files (relative_image_path<TAB>text). The
paper's three experiments map to:
    exp1: --train fca/train.tsv
    exp2: --train fca/train.tsv crohme.tsv
    exp3: --train fca/train.tsv crohme.tsv synth.tsv
Validation picks the best epoch by CER; --test reports WER/CER at the end.

Usage (Colab):
    python -m hdr.train.train_trocr --name exp3 \
        --train data_work/text/fca/train.tsv data_work/text/crohme.tsv data_work/text/synth.tsv \
        --val data_work/text/fca/val.tsv --test data_work/text/fca/test.tsv \
        --out /content/drive/MyDrive/hdr/trocr
Evaluate only:  --eval-only --model <dir with saved model> --test ...
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import jiwer
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def read_tsv(tsv: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            rel, text = line.split("\t", 1)
            rows.append((tsv.parent / rel, text))
    return rows


class TextLineDataset(Dataset):
    def __init__(self, rows, processor, max_len: int = 48):
        self.rows, self.processor, self.max_len = rows, processor, max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        # Pretrained TrOCR decodes "text </s>" after the start token (</s>),
        # so labels carry no leading <s>. Padding positions are ignored (-100).
        tok = self.processor.tokenizer
        ids = tok(text, add_special_tokens=False, truncation=True,
                  max_length=self.max_len - 1).input_ids + [tok.sep_token_id or tok.eos_token_id]
        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        labels[:len(ids)] = torch.tensor(ids)
        return {"pixel_values": pixel_values, "labels": labels, "text": text}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


def seq_loss(model, pixel_values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token cross-entropy computed here, not by the model.

    Some transformers versions (seen with 4.57) route VisionEncoderDecoder
    labels through ForCausalLMLoss, which shifts them a second time and
    misaligns every target. Building decoder inputs ourselves avoids that.
    """
    start, pad = model.generation_config.decoder_start_token_id, model.generation_config.pad_token_id
    dec_in = torch.full_like(labels, pad)
    dec_in[:, 0] = start
    dec_in[:, 1:] = labels[:, :-1]
    dec_in[dec_in == -100] = pad
    logits = model(pixel_values=pixel_values, decoder_input_ids=dec_in).logits
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(model, processor, rows, device, batch: int = 16, beams: int = 4) -> dict:
    model.eval()
    preds, refs = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        images = [Image.open(p).convert("RGB") for p, _ in chunk]
        pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pv, num_beams=beams, max_new_tokens=48)
        preds += [s.strip() for s in processor.batch_decode(out, skip_special_tokens=True)]
        refs += [t for _, t in chunk]
    # jiwer needs non-empty references
    pairs_ = [(r, p) for r, p in zip(refs, preds) if r.strip()]
    r, p = [a for a, _ in pairs_], [b for _, b in pairs_]
    return {"wer": jiwer.wer(r, p), "cer": jiwer.cer(r, p), "n": len(r),
            "samples": list(zip(r[:10], p[:10]))}


def _first(*values):
    return next((v for v in values if v is not None), None)


def configure(model, processor) -> tuple[int, int, int]:
    """Resolve the special-token ids that training and generate() need.

    Keep values the pretrained checkpoint already defines: overriding the
    decoder start token breaks a pretrained TrOCR's zero-shot output.

    Works on transformers 4.x and 5.x. In 5.x the top-level
    VisionEncoderDecoderConfig no longer has decoder_start_token_id /
    pad_token_id (reading them raises AttributeError); the values live in
    generation_config and the decoder config. generation_config is the
    single source of truth here (seq_loss and generate read it).
    """
    tok = processor.tokenizer
    cfg, gen, dec = model.config, model.generation_config, model.config.decoder
    start = _first(gen.decoder_start_token_id, getattr(dec, "decoder_start_token_id", None),
                   getattr(cfg, "decoder_start_token_id", None), tok.cls_token_id, tok.bos_token_id)
    pad = _first(gen.pad_token_id, getattr(dec, "pad_token_id", None),
                 getattr(cfg, "pad_token_id", None), tok.pad_token_id)
    eos = _first(gen.eos_token_id, getattr(dec, "eos_token_id", None),
                 getattr(cfg, "eos_token_id", None), tok.sep_token_id, tok.eos_token_id)
    gen.decoder_start_token_id, gen.pad_token_id, gen.eos_token_id = start, pad, eos
    for name, value in (("decoder_start_token_id", start), ("pad_token_id", pad),
                        ("eos_token_id", eos), ("vocab_size", dec.vocab_size)):
        try:  # 4.x reads these from the top-level config; 5.x may not accept them
            setattr(cfg, name, value)
        except (AttributeError, TypeError, ValueError):
            pass
    return start, pad, eos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp")
    ap.add_argument("--model", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--train", type=Path, nargs="*", default=[])
    ap.add_argument("--val", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/trocr"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only N rows per tsv (smoke test)")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    configure(model, processor)
    model.to(device)

    lim = (lambda rows: rows[: args.limit]) if args.limit else (lambda rows: rows)
    val_rows = lim(read_tsv(args.val)) if args.val else []
    test_rows = lim(read_tsv(args.test)) if args.test else []

    if not args.eval_only:
        sets = [TextLineDataset(lim(read_tsv(t)), processor) for t in args.train]
        train_ds = ConcatDataset(sets)
        print(f"train {len(train_ds)} ({', '.join(f'{t.name}:{len(s)}' for t, s in zip(args.train, sets))})"
              f"  val {len(val_rows)}  device {device}")
        dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = args.epochs * len(dl)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(steps, 1), pct_start=0.1)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        best, bad = float("inf"), 0
        if val_rows:
            # Baseline: the pretrained model counts as "epoch -1", so the saved
            # best model is never worse than no fine-tuning at all.
            m = evaluate(model, processor, val_rows, device)
            rec = {"epoch": -1, "val_wer": m["wer"], "val_cer": m["cer"]}
            print(rec, flush=True)
            with (out / "log.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            best = m["cer"]
            model.save_pretrained(out / "best")
            processor.save_pretrained(out / "best")
        for epoch in range(args.epochs):
            model.train()
            t0, total = time.time(), 0.0
            for b in dl:
                pv, labels = b["pixel_values"].to(device), b["labels"].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    loss = seq_loss(model, pv, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item()
            rec = {"epoch": epoch, "loss": total / max(len(dl), 1), "sec": round(time.time() - t0, 1)}
            if val_rows:
                m = evaluate(model, processor, val_rows, device)
                rec.update(val_wer=m["wer"], val_cer=m["cer"])
                score = m["cer"]
            else:
                score = rec["loss"]
            print(rec, flush=True)
            with (out / "log.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if score < best:
                best, bad = score, 0
                model.save_pretrained(out / "best")
                processor.save_pretrained(out / "best")
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stop")
                    break
        model = VisionEncoderDecoderModel.from_pretrained(out / "best").to(device)
        configure(model, processor)

    if test_rows:
        m = evaluate(model, processor, test_rows, device)
        print(f"TEST {args.name}: WER {m['wer']:.4f}  CER {m['cer']:.4f}  (n={m['n']})")
        for ref, pred in m["samples"]:
            print(f"  {ref!r:30} -> {pred!r}")
        (out / "test_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
