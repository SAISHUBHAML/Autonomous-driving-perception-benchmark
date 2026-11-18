"""Multi-camera 3D tracking with an Extended Kalman Filter over camera and LiDAR measurements.

Every keyframe (2 Hz), each of the six cameras gives 2D detections. Each detection becomes a
3D measurement of the object's position in the global frame, expressed as the polar pair
(range, bearing) seen from the camera that made it:

  bearing  comes from the 2D box centre and the camera intrinsics. It is precise.
  range    comes from LiDAR points inside the box when there are enough of them (precise),
           otherwise from the pinhole height prior (imprecise, error grows with distance).

The two sources get different measurement noise, so the filter trusts LiDAR range and camera
bearing, and leans on its motion model when only a camera range is available. Because
h(x) = (range, bearing) is nonlinear in the Cartesian state, this is an EKF, not a linear KF.

  state      x = [px, py, vx, vy] in the global frame (ego motion is handled by the frame choice)
  motion     constant velocity, white-acceleration process noise per class
  measure    z = [r, theta] from the camera position, noise fitted on the validation scene
  associate  Hungarian on Mahalanobis distance, class-aware, chi-square gate (99 %)
  manage     birth on unmatched measurements, confirmed after 2 hits, deleted after 2 misses

Scoring is in bird's-eye view against the nuScenes 3D boxes: a track matches an object of the
same class when their centres are within 2 m (the nuScenes tracking threshold). We report
CLEAR-MOT/IDF1, the centre error of matched tracks, and the velocity error of matched tracks.

Baselines with the same detections and the same scoring:
  no filter   the track position is just the latest measurement, matched by Euclidean distance
  camera-only every measurement uses the camera range, so LiDAR is never used
"""

from __future__ import annotations

from dataclasses import dataclass, field

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import lidar as L
from . import nusc as N
from .detection import Detector
from .localization import HALF_DEPTH, HEIGHT, MIN_POINTS, SHRINK, match

SCORE_THR = 0.3
MAX_RANGE = 50.0
MATCH_DIST = 2.0                       # nuScenes tracking centre-distance threshold (m)
GATE_CHI2 = 9.21                       # chi-square, 2 dof, 99 %
EUCLID_GATE = {"pedestrian": 1.5, "bicycle": 2.0, "motorcycle": 2.5}   # no-filter baseline, else 3 m
# Motion-model noise, picked on two training scenes (scene-0061, scene-0553) with the fine-tuned
# detector; the test scenes were not used.
ACCEL_STD = {"pedestrian": 0.75, "bicycle": 1.0, "motorcycle": 1.5}    # m/s^2, else 1.25 (vehicles)
INIT_VEL_STD = {"pedestrian": 0.75, "bicycle": 1.5, "motorcycle": 2.5}  # m/s, else 3
CONFIRM_HITS = 2
MAX_MISSES = 2


@dataclass
class Measurement:
    cls: str
    score: float
    p: np.ndarray            # (3,) global position estimate
    origin: np.ndarray       # (2,) camera position in the global frame (BEV)
    source: str              # "lidar" or "camera"

    @property
    def polar(self) -> np.ndarray:
        d = self.p[:2] - self.origin
        return np.array([np.hypot(*d), np.arctan2(d[1], d[0])])


@dataclass
class Noise:
    """Measurement noise: sigma_r = a + b * r (per source), sigma_theta constant (rad)."""
    r_lidar: tuple[float, float]
    r_camera: tuple[float, float]
    theta: float

    def R(self, m: Measurement) -> np.ndarray:
        a, b = self.r_lidar if m.source == "lidar" else self.r_camera
        r = m.polar[0]
        return np.diag([(a + b * r) ** 2, self.theta ** 2])


# ----------------------------------------------------------------------------- measurements

def camera_measurements(nusc, cam_tok: str, det, use_lidar: bool) -> list[Measurement]:
    """Turn one camera's detections into global-frame 3D measurements."""
    K = N.intrinsics(nusc, cam_tok)
    T = L.sensor_to_global(nusc, cam_tok)
    origin = T[:2, 3]
    uv, depth = N.lidar_in_image(nusc, cam_tok) if use_lidar else (None, None)
    out = []
    for (x1, y1, x2, y2), s, cls in zip(det.xyxy, det.conf, det.cls):
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        z, source = K[1, 1] * HEIGHT[cls] / max(y2 - y1, 1.0), "camera"
        if use_lidar:
            hw, hh = (x2 - x1) * SHRINK / 2, (y2 - y1) * SHRINK / 2
            inside = (abs(uv[0] - u) <= hw) & (abs(uv[1] - v) <= hh)
            if inside.sum() >= MIN_POINTS:
                z, source = float(np.median(depth[inside])) + HALF_DEPTH[cls], "lidar"
        p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        out.append(Measurement(cls, float(s), L.apply(T, p_cam[None])[0], origin.copy(), source))
    return out


def dedupe(ms: list[Measurement]) -> list[Measurement]:
    """Objects in the overlap of two cameras are detected twice: keep one per object."""
    order = sorted(range(len(ms)), key=lambda i: (ms[i].source != "lidar", -ms[i].score))
    kept: list[Measurement] = []
    for i in order:
        m = ms[i]
        if all(k.cls != m.cls or np.linalg.norm(k.p[:2] - m.p[:2]) > max(1.0, 0.1 * m.polar[0])
               for k in kept):
            kept.append(m)
    return kept


# ----------------------------------------------------------------------------- EKF

def _h(x: np.ndarray, origin: np.ndarray):
    dx, dy = x[0] - origin[0], x[1] - origin[1]
    r = max(np.hypot(dx, dy), 1e-3)
    z = np.array([r, np.arctan2(dy, dx)])
    H = np.array([[dx / r, dy / r, 0, 0],
                  [-dy / r ** 2, dx / r ** 2, 0, 0]])
    return z, H


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Track:
    id: int
    cls: str
    x: np.ndarray
    P: np.ndarray
    z: float
    hits: int = 1
    misses: int = 0
    age: int = 1
    history: list = field(default_factory=list)

    def predict(self, dt: float, use_filter: bool):
        if not use_filter:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = ACCEL_STD.get(self.cls, 1.25) ** 2
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q

    def innovation(self, m: Measurement, noise: Noise):
        z_hat, H = _h(self.x, m.origin)
        y = m.polar - z_hat
        y[1] = _wrap(y[1])
        S = H @ self.P @ H.T + noise.R(m)
        return y, H, S

    def update(self, m: Measurement, noise: Noise, use_filter: bool):
        if use_filter:
            y, H, S = self.innovation(m, noise)
            Kg = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + Kg @ y
            self.P = (np.eye(4) - Kg @ H) @ self.P
        else:
            self.x[:2] = m.p[:2]
        self.z = 0.7 * self.z + 0.3 * m.p[2]
        self.hits += 1
        self.misses = 0


def _init_cov(m: Measurement, noise: Noise) -> np.ndarray:
    r, th = m.polar
    J = np.array([[np.cos(th), -r * np.sin(th)], [np.sin(th), r * np.cos(th)]])
    P = np.zeros((4, 4))
    P[:2, :2] = J @ noise.R(m) @ J.T
    P[2, 2] = P[3, 3] = INIT_VEL_STD.get(m.cls, 3.0) ** 2
    return P


class MultiObjectTracker:
    def __init__(self, noise: Noise, use_filter: bool = True):
        self.noise, self.use_filter = noise, use_filter
        self.tracks: list[Track] = []
        self.next_id = 0
        self.t = None

    def _cost(self, t: Track, m: Measurement) -> float:
        if t.cls != m.cls:
            return np.inf
        if self.use_filter:
            y, _, S = t.innovation(m, self.noise)
            d2 = float(y @ np.linalg.solve(S, y))
            return d2 if d2 <= GATE_CHI2 else np.inf
        d = float(np.linalg.norm(t.x[:2] - m.p[:2]))
        return d if d <= EUCLID_GATE.get(m.cls, 3.0) else np.inf

    def step(self, timestamp_us: int, ms: list[Measurement]) -> list[Track]:
        dt = 0.0 if self.t is None else (timestamp_us - self.t) / 1e6
        self.t = timestamp_us
        for t in self.tracks:
            t.predict(dt, self.use_filter)
            t.age += 1

        matched_t, matched_m = set(), set()
        if self.tracks and ms:
            C = np.array([[self._cost(t, m) for m in ms] for t in self.tracks])
            big = 1e6
            rows, cols = linear_sum_assignment(np.where(np.isfinite(C), C, big))
            for i, j in zip(rows, cols):
                if np.isfinite(C[i, j]):
                    self.tracks[i].update(ms[j], self.noise, self.use_filter)
                    matched_t.add(i)
                    matched_m.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]
        for j, m in enumerate(ms):
            if j not in matched_m:
                x = np.r_[m.p[:2], 0.0, 0.0]
                self.tracks.append(Track(self.next_id, m.cls, x, _init_cov(m, self.noise), float(m.p[2])))
                self.next_id += 1
        return [t for t in self.tracks if t.misses == 0 and (t.hits >= CONFIRM_HITS or t.age == 1)]


# ----------------------------------------------------------------------------- data

def _detections(detector: Detector, scenes: list[str]) -> dict[str, object]:
    """Detector output for every keyframe camera image, filtered at SCORE_THR (cached per run)."""
    nusc = N.load()
    out = {}
    for tok in N.keyframe_camera_tokens(nusc, scenes):
        d = detector(N.image_path(nusc, tok))
        keep = d.conf >= SCORE_THR
        d.xyxy, d.conf, d.cls = d.xyxy[keep], d.conf[keep], [c for c, k in zip(d.cls, keep) if k]
        out[tok] = d
    return out


def _samples(nusc, scene: str):
    tok = N.scene_by_name(nusc, scene)["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        yield s
        tok = s["next"]


def estimate_noise(detector: Detector, scenes: list[str]) -> tuple[Noise, dict]:
    """Fit measurement noise from residuals of matched detections (robust MAD-based sigmas)."""
    nusc = N.load()
    res = {"lidar": [], "camera": [], "theta": []}
    for scene in scenes:
        for s in _samples(nusc, scene):
            for cam in N.CAMERAS:
                tok = s["data"][cam]
                det = detector(N.image_path(nusc, tok))
                keep = det.conf >= SCORE_THR
                det.xyxy, det.conf, det.cls = det.xyxy[keep], det.conf[keep], [c for c, k in zip(det.cls, keep) if k]
                gts = [g for g in N.gt_boxes(nusc, tok) if not g.ignore]
                if not gts or not len(det.conf):
                    continue
                T = L.sensor_to_global(nusc, tok)
                ms_cam = camera_measurements(nusc, tok, det, use_lidar=False)
                ms_lid = camera_measurements(nusc, tok, det, use_lidar=True)
                for i, j in match(gts, det):
                    g = L.apply(T, gts[j].center_cam[None])[0]
                    d = g[:2] - ms_cam[i].origin
                    r_gt, th_gt = np.hypot(*d), np.arctan2(d[1], d[0])
                    res["camera"].append((r_gt, ms_cam[i].polar[0] - r_gt))
                    res["theta"].append(_wrap(ms_cam[i].polar[1] - th_gt))
                    if ms_lid[i].source == "lidar":
                        res["lidar"].append((r_gt, ms_lid[i].polar[0] - r_gt))

    def fit(rows):
        # sigma(r) = a + b r: robust scale (1.4826 * MAD) in range bins, then a weighted line fit,
        # so the few detections whose LiDAR points hit the background do not inflate the noise
        a = np.array(rows)
        r, e = a[:, 0], a[:, 1]
        centers, sig, w = [], [], []
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, MAX_RANGE)]:
            sel = (r >= lo) & (r < hi)
            if sel.sum() >= 15:
                centers.append(np.median(r[sel]))
                sig.append(1.4826 * np.median(np.abs(e[sel])))      # about 0: bias counts as error
                w.append(np.sqrt(sel.sum()))
        W = np.diag(w)
        A = np.c_[np.ones(len(centers)), centers]
        coef = np.linalg.lstsq(W @ A, W @ np.array(sig), rcond=None)[0]
        return float(max(coef[0], 0.1)), float(max(coef[1], 0.0))

    th = np.array(res["theta"])
    noise = Noise(fit(res["lidar"]), fit(res["camera"]), float(1.4826 * np.median(np.abs(th - np.median(th)))))
    info = {"scenes": scenes, "samples": {k: len(v) for k, v in res.items()},
            "sigma_range_lidar_m": {"a": noise.r_lidar[0], "b_per_m": noise.r_lidar[1]},
            "sigma_range_camera_m": {"a": noise.r_camera[0], "b_per_m": noise.r_camera[1]},
            "sigma_bearing_rad": noise.theta}
    return noise, info


# ----------------------------------------------------------------------------- evaluation

def _gt(nusc, sample):
    """Valid and ignored GT objects of the evaluated classes at one keyframe (global BEV)."""
    visible = {}
    for cam in N.CAMERAS:
        for g in N.gt_boxes(nusc, sample["data"][cam]):
            visible[g.instance] = visible.get(g.instance, False) or not g.ignore
    ego = L.sensor_to_global(nusc, sample["data"]["LIDAR_TOP"])[:2, 3]
    valid, ignored = [], []
    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        cls = N.NUSC_TO_CLASS.get(ann["category_name"])
        if cls is None:
            continue
        xy = np.array(ann["translation"][:2])
        obj = {"cls": cls, "xy": xy, "instance": ann["instance_token"], "token": ann_tok}
        ok = visible.get(ann["instance_token"], False) and np.linalg.norm(xy - ego) < MAX_RANGE
        (valid if ok else ignored).append(obj)
    return valid, ignored, ego


def build_frames(dets: dict, scenes: list[str], use_lidar: bool) -> dict[str, list]:
    """Per scene, per keyframe: (timestamp, measurements, valid GT, ignored GT, ego position)."""
    nusc = N.load()
    frames = {}
    for scene in scenes:
        frames[scene] = []
        for s in _samples(nusc, scene):
            ms = []
            for cam in N.CAMERAS:
                ms += camera_measurements(nusc, s["data"][cam], dets[s["data"][cam]], use_lidar)
            valid, ignored, ego = _gt(nusc, s)
            frames[scene].append((s["timestamp"], dedupe(ms), valid, ignored, ego))
    return frames


def run_config(frames: dict[str, list], noise: Noise, use_filter: bool) -> dict:
    nusc = N.load()
    accs, errors, vel_errors = [], [], []
    gt_ids: dict[str, int] = {}
    for scene, seq in frames.items():
        acc = mm.MOTAccumulator(auto_id=True)
        mot = MultiObjectTracker(noise, use_filter)
        for timestamp, ms, valid, ignored, ego in seq:
            tracks = mot.step(timestamp, ms)
            tracks = [t for t in tracks if np.linalg.norm(t.x[:2] - ego) < MAX_RANGE]

            def near(t, objs):
                return min((np.linalg.norm(t.x[:2] - o["xy"]) for o in objs if o["cls"] == t.cls),
                           default=np.inf)
            # drop tracks that sit on an ignored object rather than counting them as false positives
            tracks = [t for t in tracks
                      if not (near(t, ignored) < MATCH_DIST and near(t, ignored) < near(t, valid))]

            D = np.full((len(valid), len(tracks)), np.nan)
            for a, g in enumerate(valid):
                for b, t in enumerate(tracks):
                    d = float(np.linalg.norm(g["xy"] - t.x[:2]))
                    if g["cls"] == t.cls and d <= MATCH_DIST:
                        D[a, b] = d
            ids = [gt_ids.setdefault(g["instance"], len(gt_ids)) for g in valid]
            fid = acc.update(ids, [t.id for t in tracks], D)
            # centre / velocity error of this frame's matches (the assignment motmetrics made)
            for a, g in enumerate(valid):
                if acc.last_match.get(ids[a]) != fid:
                    continue
                h = acc.m[ids[a]]
                b = next((b for b, t in enumerate(tracks) if t.id == h), None)
                if b is None or np.isnan(D[a, b]):
                    continue
                t = tracks[b]
                errors.append(D[a, b])
                if use_filter and t.hits >= CONFIRM_HITS:
                    v_gt = nusc.box_velocity(g["token"])[:2]
                    if np.all(np.isfinite(v_gt)):
                        vel_errors.append(float(np.linalg.norm(t.x[2:] - v_gt)))
        accs.append(acc)

    mh = mm.metrics.create()
    o = mh.compute_many(accs, names=list(frames), generate_overall=True,
                        metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses",
                                 "num_objects", "num_unique_objects", "mostly_tracked"]).loc["OVERALL"]
    e, v = np.array(errors), np.array(vel_errors)
    return {
        "MOTA": float(o["mota"]), "IDF1": float(o["idf1"]),
        "id_switches": int(o["num_switches"]), "false_positives": int(o["num_false_positives"]),
        "misses": int(o["num_misses"]), "gt_detections": int(o["num_objects"]),
        "gt_tracks": int(o["num_unique_objects"]), "mostly_tracked": int(o["mostly_tracked"]),
        "bev_center_error_m": {"n": len(e), "mean": float(e.mean()), "median": float(np.median(e))} if len(e) else {},
        "velocity_error_mps": ({"n": len(v), "mean": float(v.mean()), "median": float(np.median(v))}
                               if len(v) else None),
    }


CONFIGS = {
    "camera-only, no filter": (False, False),
    "camera-only, EKF": (False, True),
    "camera+LiDAR, no filter": (True, False),
    "camera+LiDAR, EKF": (True, True),
}


def detections_for(detector: Detector, scenes: list[str]) -> dict:
    """Keyframe detections at SCORE_THR, shared by the EKF tracker and the cluster labelling."""
    return _detections(detector, scenes)


def evaluate_ekf_tracking(detector: Detector, dets: dict, test_scenes: list[str], noise_scenes: list[str]) -> dict:
    noise, info = estimate_noise(detector, noise_scenes)
    out = {"measurement_noise": info, "score_threshold": SCORE_THR, "match_distance_m": MATCH_DIST,
           "max_range_m": MAX_RANGE, "configs": {}}
    frames = {use_lidar: build_frames(dets, test_scenes, use_lidar) for use_lidar in (False, True)}
    for name, (use_lidar, use_filter) in CONFIGS.items():
        out["configs"][name] = run_config(frames[use_lidar], noise, use_filter)
    return out
