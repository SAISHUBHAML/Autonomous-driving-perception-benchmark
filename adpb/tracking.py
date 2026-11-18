"""2D multi-object tracking per camera, scored with CLEAR-MOT / IDF1 against nuScenes instance ids.

Each camera of each test scene is one sequence. The tracker runs on every frame (keyframes and
~12 Hz sweeps, so motion between frames is small, as trackers expect); it is scored only on
keyframes, the only frames with annotations. Ground truth identities are nuScenes instance tokens.
Tracker outputs that overlap an ignored GT box (low visibility / no LiDAR points) are dropped
before scoring rather than counted as false positives.
"""

from __future__ import annotations

import motmetrics as mm
import numpy as np

from . import nusc as N
from .detection import Detector

MATCH_IOU = 0.5
_GT_IDS: dict[str, int] = {}   # motmetrics needs numeric ids


def _gt_id(instance_token: str) -> int:
    return _GT_IDS.setdefault(instance_token, len(_GT_IDS))


def _xywh(xyxy: np.ndarray) -> np.ndarray:
    return np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]] if len(xyxy) else np.zeros((0, 4))


def run_sequence(weights: str, imgsz: int, tracker: str, frames: list[dict], acc: mm.MOTAccumulator):
    nusc = N.load()
    det = Detector(weights, imgsz=imgsz)          # fresh model => fresh tracker state per sequence
    for sd in frames:
        out = det.track(N.image_path(nusc, sd["token"]), tracker)
        if not sd["is_key_frame"]:
            continue
        gts = N.gt_boxes(nusc, sd["token"])
        valid = [g for g in gts if not g.ignore]
        ignored = np.array([g.xyxy for g in gts if g.ignore]).reshape(-1, 4)

        hyp_xyxy = out.xyxy if out.track_id is not None else np.zeros((0, 4))
        hyp_ids = out.track_id if out.track_id is not None else np.zeros(0, int)
        hyp_cls = out.cls if out.track_id is not None else []
        if len(hyp_xyxy) and len(ignored):
            gt_valid = np.array([g.xyxy for g in valid]).reshape(-1, 4)
            best_valid = N.iou_matrix(hyp_xyxy, gt_valid).max(1) if len(gt_valid) else np.zeros(len(hyp_xyxy))
            best_ignored = N.iou_matrix(hyp_xyxy, ignored).max(1)
            keep = ~((best_ignored >= MATCH_IOU) & (best_ignored > best_valid))
            hyp_xyxy, hyp_ids = hyp_xyxy[keep], hyp_ids[keep]
            hyp_cls = [c for c, k in zip(hyp_cls, keep) if k]

        gt_xyxy = np.array([g.xyxy for g in valid]).reshape(-1, 4)
        dist = mm.distances.iou_matrix(_xywh(gt_xyxy), _xywh(hyp_xyxy), max_iou=1 - MATCH_IOU)
        for a, g in enumerate(valid):                 # class-aware: different classes never match
            for b, c in enumerate(hyp_cls):
                if g.cls != c:
                    dist[a, b] = np.nan
        acc.update([_gt_id(g.instance) for g in valid], [int(i) for i in hyp_ids], dist)


def evaluate_tracking(weights: str, scenes: list[str], tracker: str = "bytetrack.yaml",
                      imgsz: int = 960, cameras=N.CAMERAS) -> dict:
    nusc = N.load()
    accs, names = [], []
    for scene in scenes:
        for cam in cameras:
            acc = mm.MOTAccumulator(auto_id=True)
            run_sequence(weights, imgsz, tracker, N.camera_sequence(nusc, scene, cam), acc)
            accs.append(acc)
            names.append(f"{scene}/{cam}")
    mh = mm.metrics.create()
    metrics = ["mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses",
               "num_objects", "mostly_tracked", "mostly_lost", "num_unique_objects"]
    summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
    o = summary.loc["OVERALL"]
    return {
        "tracker": tracker,
        "sequences": len(accs),
        "MOTA": float(o["mota"]),
        "IDF1": float(o["idf1"]),
        "MOTP_iou_distance": float(o["motp"]),
        "id_switches": int(o["num_switches"]),
        "false_positives": int(o["num_false_positives"]),
        "misses": int(o["num_misses"]),
        "gt_detections": int(o["num_objects"]),
        "gt_tracks": int(o["num_unique_objects"]),
        "mostly_tracked": int(o["mostly_tracked"]),
        "mostly_lost": int(o["mostly_lost"]),
    }
