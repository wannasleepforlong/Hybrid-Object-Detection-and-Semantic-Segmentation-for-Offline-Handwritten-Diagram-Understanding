"""Fine-tune DeepLabv3 for binary arrow segmentation (paper section III-C).

Paper settings: DeepLabv3-ResNet50, pretrained head of 21 classes replaced by
2 classes (arrow / not arrow), images resized to 700x700, Adam, lr 1e-5,
cross-entropy. The paper ran 10,000 epochs; here training stops early when
validation arrow IoU stops improving.

Train set: FC_A train + FC_B train (528 images). Validation: FC_B val.

Usage (Colab):
    python -m hdr.train.train_deeplab --out /content/drive/MyDrive/hdr/deeplab
Resumes automatically from <out>/last.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

from hdr import config

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(gray: np.ndarray, size: int) -> torch.Tensor:
    """Grayscale uint8 image -> normalised 3x size x size tensor."""
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    img = np.repeat(img[..., None].astype(np.float32) / 255.0, 3, axis=2)
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1))


def pairs(datasets: list[str], split: str) -> list[tuple[Path, Path]]:
    out = []
    for ds in datasets:
        img_dir = config.DATASETS_ROOT / ds / split
        mask_dir = config.MASK_ROOT / ds / split
        for m in sorted(mask_dir.glob("*.png")):
            out.append((img_dir / m.name, m))
    return out


class ArrowMaskDataset(Dataset):
    def __init__(self, items: list[tuple[Path, Path]], size: int, augment: bool):
        self.items, self.size, self.augment = items, size, augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        img_p, mask_p = self.items[i]
        gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            if random.random() < 0.5:
                gray, mask = gray[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                # brightness / contrast jitter, like scans and photos
                a, b = random.uniform(0.7, 1.2), random.uniform(-25, 25)
                gray = np.clip(gray.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        x = preprocess(np.ascontiguousarray(gray), self.size)
        # Thin strokes vanish with nearest-neighbour downscaling; use area
        # interpolation and threshold instead.
        m = cv2.resize(np.ascontiguousarray(mask), (self.size, self.size), interpolation=cv2.INTER_AREA)
        y = torch.from_numpy((m > 40).astype(np.int64))
        return x, y


def build_model(pretrained: bool) -> nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, weights_backbone=None, aux_loss=None)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    inter = union = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(x)["out"].argmax(1)
        inter += ((pred == 1) & (y == 1)).sum().item()
        union += ((pred == 1) | (y == 1)).sum().item()
    return inter / max(union, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.OUT_ROOT / "deeplab")
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--arrow-weight", type=float, default=1.0,
                    help="cross-entropy weight of the arrow class (paper: plain CE = 1.0)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use only N images (smoke test)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    train_items = pairs(["fca", "fcb"], "train")
    val_items = pairs(["fcb"], "val")
    if args.limit:
        train_items, val_items = train_items[: args.limit], val_items[: max(1, args.limit // 2)]
    print(f"train {len(train_items)}  val {len(val_items)}  device {device}")

    train_dl = DataLoader(ArrowMaskDataset(train_items, args.size, True), args.batch,
                          shuffle=True, num_workers=args.workers, drop_last=len(train_items) > args.batch)
    val_dl = DataLoader(ArrowMaskDataset(val_items, args.size, False), args.batch,
                        num_workers=args.workers)

    model = build_model(not args.no_pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.arrow_weight], device=device))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start, best, bad = 0, -1.0, 0
    last = args.out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"resumed at epoch {start}, best IoU {best:.4f}")

    log = args.out / "log.jsonl"
    for epoch in range(start, args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x)["out"], y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.size(0)
        iou = evaluate(model, val_dl, device)
        rec = {"epoch": epoch, "loss": total / len(train_dl.dataset), "val_iou": iou,
               "sec": round(time.time() - t0, 1)}
        print(rec, flush=True)
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if iou > best:
            best, bad = iou, 0
            torch.save(model.state_dict(), args.out / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best": best, "bad": bad}, last)
        if bad >= args.patience:
            print(f"early stop: no IoU gain for {args.patience} epochs")
            break
    print(f"best val arrow IoU {best:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
