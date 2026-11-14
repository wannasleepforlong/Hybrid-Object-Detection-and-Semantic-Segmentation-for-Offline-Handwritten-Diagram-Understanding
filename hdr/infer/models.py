"""Load the three trained models and run them on CPU (or GPU if present).

    Detector   YOLOv7 best.pt      -> list[Detection] (boxes, classes, scores)
    Segmenter  DeepLabv3 best.pt   -> binary arrow mask at full image size
    Reader     TrOCR exp*/best     -> text of an image crop

YOLOv7 weights are pickled Python objects, so YOLOv7's own code must be
importable: it is cloned to third_party/yolov7.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from hdr import config

YOLOV7_DIR = config.ROOT / "third_party" / "yolov7"
WEIGHTS = config.ROOT / "weights" / "hdr"


@dataclass
class Detection:
    cls: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector:
    """YOLOv7 inference with YOLOv7's own letterbox and NMS."""

    def __init__(self, weights: Path, img_size: int = 640):
        if str(YOLOV7_DIR) not in sys.path:
            sys.path.insert(0, str(YOLOV7_DIR))
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        from models.experimental import attempt_load  # noqa: E402  (YOLOv7 code)
        from utils.datasets import letterbox
        from utils.general import non_max_suppression, scale_coords

        self._letterbox, self._nms, self._scale = letterbox, non_max_suppression, scale_coords
        self.device = _device()
        self.model = attempt_load(str(weights), map_location=self.device).eval()
        self.names = self.model.names
        self.img_size = img_size

    @torch.no_grad()
    def forward(self, img_bgr: np.ndarray):
        """Raw network output, before NMS (run once, apply several NMS settings)."""
        img = self._letterbox(img_bgr, self.img_size, stride=32, auto=False)[0]
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        t = torch.from_numpy(img).to(self.device).float()[None] / 255.0
        return self.model(t)[0], t.shape[2:], img_bgr.shape

    def nms(self, raw, conf: float, iou: float) -> list[Detection]:
        pred, net_shape, img_shape = raw
        det = self._nms(pred.clone(), conf, iou)[0]
        if det is None or not len(det):
            return []
        det[:, :4] = self._scale(net_shape, det[:, :4], img_shape).round()
        return [Detection(self.names[int(c)], tuple(map(float, xyxy)), float(p))
                for *xyxy, p, c in det.tolist()]

    def __call__(self, img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.9) -> list[Detection]:
        """`iou` is kept high: YOLO's NMS only removes near-identical boxes;
        the paper's class-specific duplicate removal happens in postprocess."""
        return self.nms(self.forward(img_bgr), conf, iou)


class Segmenter:
    """DeepLabv3 arrow / not-arrow segmentation."""

    def __init__(self, weights: Path, size: int = 700):
        from hdr.train.train_deeplab import build_model
        self.device = _device()
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.to(self.device).eval()
        self.size = size

    @torch.no_grad()
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        from hdr.train.train_deeplab import preprocess
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        x = preprocess(gray, self.size)[None].to(self.device)
        prob = self.model(x)["out"].softmax(1)[0, 1].cpu().numpy()
        prob = cv2.resize(prob, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        return (prob > 0.5).astype(np.uint8)


class Reader:
    """TrOCR handwritten text recognition."""

    def __init__(self, model_dir: Path, base: str = "microsoft/trocr-base-handwritten"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from hdr.train.train_trocr import configure
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_dir)
        except Exception:
            # saved by a newer transformers: the tokenizer and image settings are
            # unchanged by fine-tuning, so take them from the base model
            self.processor = TrOCRProcessor.from_pretrained(base)
        self.device = _device()
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(self.device).eval()
        configure(self.model, self.processor)

    @torch.no_grad()
    def __call__(self, crops: list[np.ndarray], beams: int = 4) -> list[str]:
        from PIL import Image
        if not crops:
            return []
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        pv = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        out = self.model.generate(pv, num_beams=beams, max_new_tokens=48)
        return [s.strip() for s in self.processor.batch_decode(out, skip_special_tokens=True)]


def default_weights(dataset: str) -> dict[str, Path]:
    """Weight paths inside weights/hdr for a dataset family (fca, fcb or fa)."""
    state_best = "exp3"
    st = WEIGHTS / "STATE.json"
    if st.exists():
        import json
        state_best = json.loads(st.read_text()).get("trocr", {}).get("best_exp", state_best)
    return {
        "yolo": WEIGHTS / "yolov7_runs" / dataset / "weights" / "best.pt",
        "deeplab": WEIGHTS / "deeplab" / "best.pt",
        "trocr": WEIGHTS / "trocr" / state_best / "best",
    }
