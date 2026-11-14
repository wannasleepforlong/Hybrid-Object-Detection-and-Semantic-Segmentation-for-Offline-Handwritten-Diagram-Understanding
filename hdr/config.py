"""Shared paths, class lists and thresholds.

Paths can be overridden with environment variables (used on Colab):
HDR_DATASETS_ROOT, HDR_OUT_ROOT, HDR_YOLO_DATA_ROOT, HDR_WORK_ROOT.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if os.environ.get(env) else default


DATASETS_ROOT = _path("HDR_DATASETS_ROOT", ROOT / "handwritten-diagram-datasets" / "datasets")
OUT_ROOT = _path("HDR_OUT_ROOT", ROOT / "out")
YOLO_DATA_ROOT = _path("HDR_YOLO_DATA_ROOT", ROOT / "data_yolo")
# Generated training data for segmentation and OCR.
WORK_ROOT = _path("HDR_WORK_ROOT", ROOT / "data_work")
MASK_ROOT = WORK_ROOT / "arrow_masks"
TEXT_ROOT = WORK_ROOT / "text"

# Detector classes per dataset family. Order defines YOLO class ids.
FLOWCHART_CLASSES = [
    "arrow", "connection", "data", "decision", "process", "terminator", "text",
    "arrow_head", "arrow_tail",
]
FA_CLASSES = ["arrow", "state", "final state", "text", "arrow_head", "arrow_tail"]

DATASET_CLASSES = {
    "fca": FLOWCHART_CLASSES,
    "fcb": FLOWCHART_CLASSES,
    "fcb_scan": FLOWCHART_CLASSES,
    "fa": FA_CLASSES,
}

# COCO arrow keypoints are stored as [x0, y0, v0, x1, y1, v1].
# The category metadata lists them as ['head', 'tail'], but that order is
# misleading: keypoint 0 lies at the `arrow_prev` symbol (where the arrow
# starts) and keypoint 1 lies at `arrow_next`, where the drawn arrowhead is.
# Confirmed by hdr.data.check_conventions (FC_A 3092/3092, FC_B 4568/4569)
# and by visual overlays. `arrow_prev` / `arrow_next` hold annotation `id`s.
ARROW_TAIL_KP_INDEX = 0
ARROW_HEAD_KP_INDEX = 1

# Side of the square arrow_head / arrow_tail boxes, as a fraction of the
# image's longer side. Not given in the paper; tune on validation data.
ENDPOINT_BOX_FRAC = 0.03
ENDPOINT_BOX_MIN_PX = 16

# Post-processing thresholds (paper section III-C).
GENERAL_THRESHOLD = 0.50
ARROW_THRESHOLD = 0.60

# Arrow mask stroke width in pixels (paper section III-C).
ARROW_MASK_WIDTH = 6
