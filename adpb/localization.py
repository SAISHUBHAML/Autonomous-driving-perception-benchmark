"""3D localization of detected objects: monocular geometry vs. LiDAR depth, scored against nuScenes GT.

For every detection matched to a ground-truth box (same class, IoU >= 0.5), we estimate the
object's 3D center in the camera frame by back-projecting the 2D box center at an estimated depth:

  camera   depth = f_y * H_class / box_height_px      (pinhole model with a per-class height prior)
  lidar    depth = median depth of LIDAR_TOP points falling in the central part of the box,
                   plus a per-class half-extent offset (points hit the near surface, not the center)
  fused    lidar depth when the box contains >= MIN_POINTS points, otherwise the camera depth

and report the error against the annotated 3D box center.
"""

from __future__ import annotations

import numpy as np

from . import nusc as N
from .detection import Detector

# Typical object heights (m) for the pinhole depth prior, and half of the typical extent seen
# along the viewing ray (m) for correcting LiDAR surface depth to the object center.
HEIGHT = {"car": 1.6, "truck": 2.8, "bus": 3.3, "motorcycle": 1.4, "bicycle": 1.2, "pedestrian": 1.75}
HALF_DEPTH = {"car": 1.2, "truck": 1.6, "bus": 1.8, "motorcycle": 0.5, "bicycle": 0.5, "pedestrian": 0.3}
MIN_POINTS = 3
SHRINK = 0.5      # use the central 50% (per side) of the box to avoid background points at its edges
MATCH_IOU = 0.5
RANGES = [(0, 20), (20, 40), (40, 1e9)]


def match(gts: list[N.GTBox], det, iou_thr=MATCH_IOU):
    """Greedy one-to-one matching by confidence, same class only."""
    gt_xyxy = np.array([g.xyxy for g in gts]) if gts else np.zeros((0, 4))
    iou = N.iou_matrix(det.xyxy, gt_xyxy)
    used, pairs = set(), []
    for i in np.argsort(-det.conf):
        best, best_iou = None, iou_thr
        for j, g in enumerate(gts):
            if j in used or g.cls != det.cls[i]:
                continue
            if iou[i, j] >= best_iou:
                best, best_iou = j, iou[i, j]
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def _backproject(u, v, z, K):
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def _summary(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {"n": 0}
    return {"n": int(len(err)), "mean": float(err.mean()), "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90))}


def evaluate_localization(detector: Detector, sd_tokens: list[str], score_thr: float = 0.25) -> dict:
    nusc = N.load()
    rows = []   # one row per matched (non-ignored) GT box
    for tok in sd_tokens:
        gts = [g for g in N.gt_boxes(nusc, tok) if not g.ignore]
        det = detector(N.image_path(nusc, tok))
        keep = det.conf >= score_thr
        det.xyxy, det.conf, det.cls = det.xyxy[keep], det.conf[keep], [c for c, k in zip(det.cls, keep) if k]
        if not gts or len(det.conf) == 0:
            continue
        K = N.intrinsics(nusc, tok)
        uv, depth = N.lidar_in_image(nusc, tok)
        for i, j in match(gts, det):
            x1, y1, x2, y2 = det.xyxy[i]
            g, cls = gts[j], det.cls[i]
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            z_cam = K[1, 1] * HEIGHT[cls] / max(y2 - y1, 1.0)

            hw, hh = (x2 - x1) * SHRINK / 2, (y2 - y1) * SHRINK / 2
            inside = (abs(uv[0] - u) <= hw) & (abs(uv[1] - v) <= hh)
            n_pts = int(inside.sum())
            z_lidar = float(np.median(depth[inside])) + HALF_DEPTH[cls] if n_pts >= MIN_POINTS else None
            z_fused = z_lidar if z_lidar is not None else z_cam

            gt_c = g.center_cam
            rows.append({
                "cls": cls, "gt_depth": float(gt_c[2]), "lidar_points": n_pts,
                "camera": float(np.linalg.norm(_backproject(u, v, z_cam, K) - gt_c)),
                "lidar": float(np.linalg.norm(_backproject(u, v, z_lidar, K) - gt_c)) if z_lidar else None,
                "fused": float(np.linalg.norm(_backproject(u, v, z_fused, K) - gt_c)),
            })

    def errs(method, subset):
        return np.array([r[method] for r in subset if r[method] is not None])

    with_lidar = [r for r in rows if r["lidar"] is not None]
    out = {
        "matched_objects": len(rows),
        "objects_with_lidar_depth": len(with_lidar),
        "center_error_m": {m: _summary(errs(m, rows)) for m in ("camera", "fused")},
        "center_error_m_where_lidar_available": {m: _summary(errs(m, with_lidar))
                                                 for m in ("camera", "lidar")},
        "by_range": {},
        "by_class": {},
    }
    for lo, hi in RANGES:
        sub = [r for r in rows if lo <= r["gt_depth"] < hi]
        key = f"{lo}-{hi:g}m" if hi < 1e9 else f">{lo}m"
        out["by_range"][key] = {m: _summary(errs(m, sub)) for m in ("camera", "fused")}
    for c in N.CLASSES:
        sub = [r for r in rows if r["cls"] == c]
        out["by_class"][c] = {m: _summary(errs(m, sub)) for m in ("camera", "fused")}
    return out
