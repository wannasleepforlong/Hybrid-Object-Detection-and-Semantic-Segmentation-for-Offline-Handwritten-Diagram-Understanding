"""Load the COCO-format diagram datasets into simple Python structures."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from hdr.config import DATASETS_ROOT


@dataclass
class Annotation:
    id: int
    aid: int | None
    category: str
    bbox: tuple[float, float, float, float]  # x, y, w, h
    keypoints: list[float] | None = None
    waypoints: list[list[float]] | None = None
    arrow_prev: int | None = None
    arrow_next: int | None = None
    text: str | None = None


@dataclass
class ImageRecord:
    id: int
    file_name: str
    path: Path
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)


def load_split(dataset: str, split: str, root: Path = DATASETS_ROOT) -> list[ImageRecord]:
    """Return all images of one split with their annotations attached."""
    ds_dir = root / dataset
    data = json.loads((ds_dir / f"{split}.json").read_text(encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    images = {
        im["id"]: ImageRecord(
            id=im["id"],
            file_name=im["file_name"],
            path=ds_dir / split / im["file_name"],
            width=im["width"],
            height=im["height"],
        )
        for im in data["images"]
    }
    for a in data["annotations"]:
        images[a["image_id"]].annotations.append(
            Annotation(
                id=a["id"],
                aid=a.get("aid"),
                category=a.get("category") or cat_names[a["category_id"]],
                bbox=tuple(a["bbox"]),
                keypoints=a.get("keypoints"),
                waypoints=a.get("waypoints"),
                arrow_prev=a.get("arrow_prev"),
                arrow_next=a.get("arrow_next"),
                text=a.get("text"),
            )
        )
    return [images[k] for k in sorted(images)]


def available_splits(dataset: str, root: Path = DATASETS_ROOT) -> list[str]:
    return [s for s in ("train", "val", "test") if (root / dataset / f"{s}.json").exists()]
