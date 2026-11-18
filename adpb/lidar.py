"""LiDAR-only object segmentation: ground removal + DBSCAN clustering, scored against nuScenes 3D boxes.

For one LIDAR_TOP keyframe sweep (32 beams, sensor frame: x forward-ish, y left, z up):

  1. crop to MAX_RANGE metres and drop returns from the ego vehicle itself
  2. ground removal: RANSAC plane fit on the low points of each range ring, then drop every
     point less than GROUND_TOL above its ring's plane (roads are not one flat plane)
  3. drop points more than MAX_HEIGHT above the ground (tree canopies, overhangs, signs)
  4. DBSCAN in the ground (x, y) plane; clusters that are too flat or too large are discarded

The clustering is class-agnostic. It is scored by whether each annotated object of the evaluated
classes contains a cluster (recall), and whether each cluster lies on any annotated object of
any nuScenes category (precision). nuScenes does not annotate walls, poles or vegetation, so
precision is a lower bound on how many clusters are real objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from sklearn.cluster import DBSCAN

from . import nusc as N

MAX_RANGE = 50.0
EGO_BOX = (-1.5, 1.5, -3.0, 3.0)    # |y| and x limits of the ego car in the lidar frame (m)
RINGS = [0, 10, 20, 30, 40, MAX_RANGE + 1]
GROUND_TOL = 0.25
MAX_HEIGHT = 4.0
MIN_CLUSTER_HEIGHT = 0.3
MAX_CLUSTER_EXTENT = 15.0

# DBSCAN parameters. Chosen on the training scenes with scripts/tune_clustering.py, never on test.
EPS = 0.3
MIN_SAMPLES = 5


# ----------------------------------------------------------------------------- geometry

def sensor_to_global(nusc, sd_token: str) -> np.ndarray:
    """4x4 transform from a sensor frame (at its own timestamp) to the global frame."""
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    t_cs = np.eye(4)
    t_cs[:3, :3], t_cs[:3, 3] = Quaternion(cs["rotation"]).rotation_matrix, cs["translation"]
    t_ep = np.eye(4)
    t_ep[:3, :3], t_ep[:3, 3] = Quaternion(ep["rotation"]).rotation_matrix, ep["translation"]
    return t_ep @ t_cs


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def load_points(nusc, lidar_sd_token: str) -> np.ndarray:
    pc = LidarPointCloud.from_file(str(N.DATAROOT / nusc.get("sample_data", lidar_sd_token)["filename"]))
    return pc.points[:3].T.copy()


# ----------------------------------------------------------------------------- segmentation

def _ransac_plane(p: np.ndarray, rng: np.random.Generator, iters=60, tol=0.15):
    """Fit z = a*x + b*y + c to (N, 3) points; returns (a, b, c) or None."""
    if len(p) < 30:
        return None
    best, best_n = None, 0
    for _ in range(iters):
        s = p[rng.choice(len(p), 3, replace=False)]
        A = np.c_[s[:, :2], np.ones(3)]
        try:
            coef = np.linalg.solve(A, s[:, 2])
        except np.linalg.LinAlgError:
            continue
        if np.hypot(coef[0], coef[1]) > 0.15:          # steeper than ~8.5 deg: not a road
            continue
        n = int((np.abs(p[:, :2] @ coef[:2] + coef[2] - p[:, 2]) < tol).sum())
        if n > best_n:
            best, best_n = coef, n
    if best is None:
        return None
    inl = np.abs(p[:, :2] @ best[:2] + best[2] - p[:, 2]) < tol      # least-squares refit on inliers
    return np.linalg.lstsq(np.c_[p[inl, :2], np.ones(inl.sum())], p[inl, 2], rcond=None)[0]


def height_above_ground(pts: np.ndarray, seed: int = 0) -> np.ndarray:
    """Per-point height above a piecewise (per range ring) RANSAC ground plane."""
    rng = np.random.default_rng(seed)
    r = np.hypot(pts[:, 0], pts[:, 1])
    h = np.full(len(pts), np.nan)
    prev = np.array([0.0, 0.0, np.percentile(pts[:, 2], 5)])
    for lo, hi in zip(RINGS[:-1], RINGS[1:]):
        ring = (r >= lo) & (r < hi)
        if not ring.any():
            continue
        p = pts[ring]
        low = p[p[:, 2] < np.percentile(p[:, 2], 40) + 0.3]    # ground candidates: the lowest points
        coef = _ransac_plane(low, rng)
        coef = prev if coef is None else coef
        prev = coef
        h[ring] = p[:, 2] - (p[:, :2] @ coef[:2] + coef[2])
    return h


@dataclass
class Segmentation:
    points: np.ndarray        # (N, 3) all cropped points, lidar frame
    height: np.ndarray        # (N,) height above ground
    labels: np.ndarray        # (N,) cluster id, -1 for ground / noise / discarded
    clusters: list[dict]      # per cluster: id, n, center (3,), min (3,), max (3,)


def segment(pts: np.ndarray, eps: float = EPS, min_samples: int = MIN_SAMPLES) -> Segmentation:
    r = np.hypot(pts[:, 0], pts[:, 1])
    ego = (np.abs(pts[:, 1]) < EGO_BOX[1]) & (pts[:, 0] > EGO_BOX[2]) & (pts[:, 0] < EGO_BOX[3])
    pts = pts[(r < MAX_RANGE) & ~ego]
    h = height_above_ground(pts)
    obj = (h > GROUND_TOL) & (h < MAX_HEIGHT)
    labels = np.full(len(pts), -1)
    clusters = []
    idx = np.flatnonzero(obj)
    if len(idx) >= min_samples:
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts[idx, :2])
        for k in range(lab.max() + 1):
            members = idx[lab == k]
            p = pts[members]
            lo, hi = p.min(0), p.max(0)
            if h[members].max() - max(h[members].min(), 0) < MIN_CLUSTER_HEIGHT:
                continue
            if np.hypot(*(hi - lo)[:2]) > MAX_CLUSTER_EXTENT:
                continue
            cid = len(clusters)
            labels[members] = cid
            clusters.append({"id": cid, "n": len(members), "center": (lo + hi) / 2, "min": lo, "max": hi})
    return Segmentation(pts, h, labels, clusters)


# ----------------------------------------------------------------------------- camera labels

def project(nusc, lidar_tok: str, cam_tok: str, pts: np.ndarray):
    """Project lidar-frame points into a camera image: (uv (N, 2), in_image mask (N,))."""
    T = np.linalg.inv(sensor_to_global(nusc, cam_tok)) @ sensor_to_global(nusc, lidar_tok)
    p = apply(T, pts)
    K = N.intrinsics(nusc, cam_tok)
    sd = nusc.get("sample_data", cam_tok)
    z = np.maximum(p[:, 2], 1e-6)
    uv = np.c_[K[0, 0] * p[:, 0] / z + K[0, 2], K[1, 1] * p[:, 1] / z + K[1, 2]]
    ok = (p[:, 2] > 1.0) & (uv[:, 0] >= 0) & (uv[:, 0] < sd["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < sd["height"])
    return uv, ok


def label_clusters(nusc, sample: dict, seg: Segmentation, dets: dict, min_points: int = 3) -> dict[int, tuple]:
    """Give each cluster the class of the camera detection that contains most of its points.

    For every detection (any camera), the cluster with the most projected points inside the box
    (at least `min_points`) is claimed by it. A cluster claimed by several detections keeps the
    highest-scoring one. Returns {cluster id: (class, score)}.
    """
    lidar_tok = sample["data"]["LIDAR_TOP"]
    claimed: dict[int, tuple] = {}
    on = seg.labels >= 0
    if not on.any():
        return claimed
    for cam in N.CAMERAS:
        cam_tok = sample["data"][cam]
        d = dets[cam_tok]
        if not len(d.conf):
            continue
        uv, ok = project(nusc, lidar_tok, cam_tok, seg.points)
        sel = ok & on
        uv_s, lab_s = uv[sel], seg.labels[sel]
        for (x1, y1, x2, y2), score, cls in zip(d.xyxy, d.conf, d.cls):
            inside = (uv_s[:, 0] >= x1) & (uv_s[:, 0] <= x2) & (uv_s[:, 1] >= y1) & (uv_s[:, 1] <= y2)
            if inside.sum() < min_points:
                continue
            ids, counts = np.unique(lab_s[inside], return_counts=True)
            k = int(ids[np.argmax(counts)])
            if counts.max() >= min_points and (k not in claimed or claimed[k][1] < score):
                claimed[k] = (cls, float(score))
    return claimed


# ----------------------------------------------------------------------------- evaluation

def _in_box_bev(xy: np.ndarray, box, margin: float) -> np.ndarray:
    """Which of the (M, 2) points fall inside the box footprint enlarged by `margin` metres."""
    R = box.rotation_matrix[:2, :2]
    local = (xy - box.center[:2]) @ R              # rotate into the box frame
    w, l = box.wlh[0] / 2 + margin, box.wlh[1] / 2 + margin
    return (np.abs(local[:, 0]) <= l) & (np.abs(local[:, 1]) <= w)


def evaluate_scene_clusters(scene_names: list[str], eps: float = EPS, min_samples: int = MIN_SAMPLES,
                            margin: float = 0.5, min_gt_points: int = 5, dets: dict | None = None) -> dict:
    """Recall / precision / centre error of DBSCAN clusters against the annotated 3D boxes.

    With `dets` (camera detections per keyframe camera token), also scores the camera-labelled
    clusters as a class-aware 3D detector: a GT object is found if a cluster of its class lies on
    it, and a labelled cluster is correct if it lies on an annotated object of its class.
    """
    nusc = N.load()
    rows, n_clusters, n_on_object = [], 0, 0
    n_labelled, n_labelled_ok = 0, 0
    for name in scene_names:
        tok = N.scene_by_name(nusc, name)["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            lidar_tok = sample["data"]["LIDAR_TOP"]
            seg = segment(load_points(nusc, lidar_tok), eps, min_samples)
            labels = label_clusters(nusc, sample, seg, dets) if dets is not None else {}
            _, boxes, _ = nusc.get_sample_data(lidar_tok)          # boxes in the lidar frame
            cen = np.array([c["center"][:2] for c in seg.clusters]).reshape(-1, 2)
            on_object = np.zeros(len(cen), bool)
            on_same_class = np.zeros(len(cen), bool)
            cluster_cls = np.array([labels[k][0] if k in labels else "" for k in range(len(cen))], dtype=object)
            box_cls = []
            for box in boxes:
                ann = nusc.get("sample_annotation", box.token)
                box_cls.append(N.NUSC_TO_CLASS.get(ann["category_name"]))
                if len(cen):
                    inb = _in_box_bev(cen, box, margin)
                    on_object |= inb
                    on_same_class |= inb & (cluster_cls == box_cls[-1])
            n_clusters += len(cen)
            n_on_object += int(on_object.sum())
            n_labelled += len(labels)
            n_labelled_ok += int(on_same_class.sum())

            for box, cls in zip(boxes, box_cls):
                ann = nusc.get("sample_annotation", box.token)
                dist = float(np.hypot(*box.center[:2]))
                if cls is None or dist >= MAX_RANGE or ann["num_lidar_pts"] < min_gt_points:
                    continue
                hit = _in_box_bev(cen, box, margin) if len(cen) else np.zeros(0, bool)
                err = None
                if hit.any():
                    err = float(np.min(np.linalg.norm(cen[hit] - box.center[:2], axis=1)))
                hit_cls = hit & (cluster_cls == cls)
                rows.append({"cls": cls, "range": dist, "found": bool(hit.any()), "n_clusters": int(hit.sum()),
                             "bev_error": err, "found_labelled": bool(hit_cls.any())})
            tok = sample["next"]

    def recall(sub, key="found"):
        return float(np.mean([r[key] for r in sub])) if sub else None

    def f(p, r, beta):
        return (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p and r else 0.0

    found = [r for r in rows if r["found"]]
    errs = np.array([r["bev_error"] for r in found])
    p = n_on_object / n_clusters if n_clusters else None
    out = {
        "params": {"eps": eps, "min_samples": min_samples, "max_range_m": MAX_RANGE,
                   "min_gt_lidar_points": min_gt_points, "match_margin_m": margin},
        "sweeps": sum(N.scene_by_name(nusc, n)["nbr_samples"] for n in scene_names),
        "gt_objects": len(rows),
        "clusters": n_clusters,
        "recall": recall(rows),
        "precision_on_annotated_objects": p,
        "f1": f(p, recall(rows), 1),
        "f2": f(p, recall(rows), 2),
        "oversegmented_fraction": float(np.mean([r["n_clusters"] > 1 for r in found])) if found else None,
        "bev_center_error_m": {"median": float(np.median(errs)), "mean": float(errs.mean())} if len(errs) else {},
        "recall_by_class": {c: recall([r for r in rows if r["cls"] == c]) for c in N.CLASSES},
        "recall_by_range": {f"{lo}-{hi:g}m": recall([r for r in rows if lo <= r["range"] < hi])
                            for lo, hi in [(0, 20), (20, 40), (40, MAX_RANGE)]},
        "gt_by_class": {c: sum(r["cls"] == c for r in rows) for c in N.CLASSES},
    }
    if dets is not None:
        pl = n_labelled_ok / n_labelled if n_labelled else None
        out["camera_labelled"] = {
            "clusters": n_labelled,
            "recall": recall(rows, "found_labelled"),
            "precision_same_class": pl,
            "f1": f(pl, recall(rows, "found_labelled"), 1),
            "recall_by_class": {c: recall([r for r in rows if r["cls"] == c], "found_labelled") for c in N.CLASSES},
            "recall_by_range": {f"{lo}-{hi:g}m": recall([r for r in rows if lo <= r["range"] < hi], "found_labelled")
                                for lo, hi in [(0, 20), (20, 40), (40, MAX_RANGE)]},
        }
    return out
