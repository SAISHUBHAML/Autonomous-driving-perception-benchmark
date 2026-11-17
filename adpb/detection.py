"""2D detection: a YOLO wrapper that speaks our class set, and COCO-style mAP against nuScenes GT."""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from . import nusc as N


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Detections:
    xyxy: np.ndarray   # (N, 4)
    conf: np.ndarray   # (N,)
    cls: list[str]     # names from N.CLASSES
    track_id: np.ndarray | None = None


class Detector:
    """YOLO model whose output classes are mapped onto N.CLASSES.

    Works for both the COCO-pretrained checkpoint (person -> pedestrian, etc.) and a
    checkpoint fine-tuned on nuScenes with N.CLASSES as its label set.
    """

    def __init__(self, weights: str, imgsz: int = 960, conf: float = 0.001, device: str | None = None):
        self.weights, self.imgsz, self.conf = weights, imgsz, conf
        self.device = device or default_device()
        self.model = YOLO(weights)
        names = self.model.names
        self.id_to_cls = {}
        for i, name in names.items():
            if name in N.CLASSES:
                self.id_to_cls[i] = name
            elif name in N.COCO_TO_CLASS:
                self.id_to_cls[i] = N.COCO_TO_CLASS[name]
        if not self.id_to_cls:
            raise ValueError(f"{weights} has no classes that map to {N.CLASSES}")
        self.class_ids = sorted(self.id_to_cls)

    def _convert(self, result) -> Detections:
        b = result.boxes
        ids = b.cls.cpu().numpy().astype(int)
        keep = np.isin(ids, self.class_ids)
        track = b.id.cpu().numpy().astype(int)[keep] if b.id is not None else None
        return Detections(b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep],
                          [self.id_to_cls[i] for i in ids[keep]], track)

    def __call__(self, image_path: str) -> Detections:
        r = self.model.predict(image_path, imgsz=self.imgsz, conf=self.conf, classes=self.class_ids,
                               device=self.device, verbose=False)[0]
        return self._convert(r)

    def track(self, image_path: str, tracker: str) -> Detections:
        r = self.model.track(image_path, imgsz=self.imgsz, classes=self.class_ids, persist=True,
                             tracker=tracker, device=self.device, verbose=False)[0]
        return self._convert(r)

    def benchmark_latency(self, image_paths: list[str], warmup: int = 10) -> dict:
        """Detector-only latency (preprocess + inference + NMS), excluding warm-up frames."""
        import cv2
        images = [cv2.imread(p) for p in image_paths[: warmup + 100]]
        for img in images[:warmup]:
            self.model.predict(img, imgsz=self.imgsz, conf=0.25, device=self.device, verbose=False)
        times = []
        for img in images[warmup:]:
            t0 = time.perf_counter()
            self.model.predict(img, imgsz=self.imgsz, conf=0.25, device=self.device, verbose=False)
            if self.device == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return {"device": self.device, "imgsz": self.imgsz, "frames": len(times),
                "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95))}


def evaluate_detection(detector: Detector, sd_tokens: list[str]) -> dict:
    """COCO-style AP over keyframe camera images. GT marked `ignore` is scored as a crowd region."""
    nusc = N.load()
    cat_id = {c: i + 1 for i, c in enumerate(N.CLASSES)}
    images, anns, dets = [], [], []
    for img_id, tok in enumerate(sd_tokens, start=1):
        sd = nusc.get("sample_data", tok)
        images.append({"id": img_id, "width": sd["width"], "height": sd["height"]})
        for g in N.gt_boxes(nusc, tok):
            x1, y1, x2, y2 = g.xyxy.tolist()
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat_id[g.cls],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                         "iscrowd": int(g.ignore)})
        d = detector(N.image_path(nusc, tok))
        for (x1, y1, x2, y2), s, c in zip(d.xyxy.tolist(), d.conf.tolist(), d.cls):
            dets.append({"image_id": img_id, "category_id": cat_id[c],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i, "name": c} for c, i in cat_id.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 300]   # up to ~100 objects per image in busy scenes
        ev.evaluate()
        ev.accumulate()

    # precision: [T(iou), R, K(class), A(area), M(maxDets)]; area=all, maxDets=300
    prec = ev.eval["precision"][:, :, :, 0, -1]
    rec = ev.eval["recall"][:, :, 0, -1]          # [T, K]
    per_class = {}
    for k, c in enumerate(N.CLASSES):
        p = prec[:, :, k]
        n_gt = sum(1 for a in anns if a["category_id"] == cat_id[c] and not a["iscrowd"])
        per_class[c] = {
            "num_gt": n_gt,
            "AP50": float(np.mean(p[0][p[0] > -1])) if (p[0] > -1).any() else None,
            "AP50_95": float(np.mean(p[p > -1])) if (p > -1).any() else None,
        }
    return {
        "images": len(images),
        "gt_boxes": sum(1 for a in anns if not a["iscrowd"]),
        "ignored_gt_boxes": sum(a["iscrowd"] for a in anns),
        "mAP50": float(np.mean(prec[0][prec[0] > -1])),
        "mAP50_95": float(np.mean(prec[prec > -1])),
        "recall50": float(np.mean(rec[0][rec[0] > -1])),
        "per_class": per_class,
    }
