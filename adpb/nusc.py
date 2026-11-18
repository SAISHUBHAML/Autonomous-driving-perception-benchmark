"""nuScenes access: scene splits, class mapping, 2D ground truth and LiDAR projection.

Everything that reads the dataset goes through this module, so every metric in
the project is computed against the official nuScenes annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / "datasets" / "nuscenes"
VERSION = "v1.0-mini"

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# Evaluated classes: the nuScenes categories that have a direct COCO counterpart,
# so the pretrained COCO detector and a fine-tuned detector are scored on the same task.
CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]

NUSC_TO_CLASS = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
}

COCO_TO_CLASS = {"car": "car", "truck": "truck", "bus": "bus",
                 "motorcycle": "motorcycle", "bicycle": "bicycle", "person": "pedestrian"}

# Scene-level split. nuScenes-mini ships only mini_train (8 scenes) and mini_val (2 scenes).
# We hold one train scene out for model selection during fine-tuning, and use the official
# mini_val scenes purely as the test set, so no test scene is ever seen during training.
HOLDOUT_SCENE = "scene-0655"


def splits() -> dict[str, list[str]]:
    s = create_splits_scenes()
    train = [n for n in s["mini_train"] if n != HOLDOUT_SCENE]
    return {"train": train, "val": [HOLDOUT_SCENE], "test": list(s["mini_val"])}


@lru_cache(maxsize=1)
def load() -> NuScenes:
    if not (DATAROOT / VERSION).exists():
        raise FileNotFoundError(
            f"{DATAROOT / VERSION} not found. Run scripts/download_nuscenes.sh first.")
    return NuScenes(VERSION, str(DATAROOT), verbose=False)


@dataclass
class GTBox:
    cls: str
    xyxy: np.ndarray          # 2D box in pixels, clipped to the image
    center_cam: np.ndarray    # 3D box center in the camera frame (x right, y down, z forward), metres
    instance: str             # nuScenes instance token = persistent track id
    ignore: bool              # low visibility or no LiDAR/radar points: neither rewarded nor penalised


def scene_by_name(nusc: NuScenes, name: str) -> dict:
    return next(s for s in nusc.scene if s["name"] == name)


def keyframe_camera_tokens(nusc: NuScenes, scene_names: list[str], cameras=CAMERAS) -> list[str]:
    """sample_data tokens of every keyframe camera image in the given scenes."""
    tokens = []
    for name in scene_names:
        tok = scene_by_name(nusc, name)["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            tokens += [sample["data"][c] for c in cameras]
            tok = sample["next"]
    return tokens


def camera_sequence(nusc: NuScenes, scene_name: str, camera: str) -> list[dict]:
    """All frames (keyframes + ~12 Hz sweeps) of one camera in one scene, in time order."""
    first = nusc.get("sample", scene_by_name(nusc, scene_name)["first_sample_token"])
    sd = nusc.get("sample_data", first["data"][camera])
    while sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])
    seq = []
    while sd:
        seq.append(sd)
        sd = nusc.get("sample_data", sd["next"]) if sd["next"] else None
    return seq


def image_path(nusc: NuScenes, sd_token: str) -> str:
    return str(DATAROOT / nusc.get("sample_data", sd_token)["filename"])


def intrinsics(nusc: NuScenes, sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    return np.array(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])["camera_intrinsic"])


def gt_boxes(nusc: NuScenes, sd_token: str) -> list[GTBox]:
    """2D ground truth for one keyframe camera image, from the projected 3D annotations."""
    sd = nusc.get("sample_data", sd_token)
    assert sd["is_key_frame"], "ground truth exists only for keyframes"
    w, h = sd["width"], sd["height"]
    _, boxes, K = nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)
    out = []
    for box in boxes:
        ann = nusc.get("sample_annotation", box.token)
        cls = NUSC_TO_CLASS.get(ann["category_name"])
        if cls is None:
            continue
        corners = box.corners()
        if (corners[2] <= 0.1).any():      # partly behind the camera: projection is undefined
            continue
        pts = view_points(corners, np.asarray(K), normalize=True)[:2]
        x1, y1 = np.clip(pts.min(1), 0, [w, h])
        x2, y2 = np.clip(pts.max(1), 0, [w, h])
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        ignore = (int(ann["visibility_token"]) < 2) or (ann["num_lidar_pts"] + ann["num_radar_pts"] == 0)
        out.append(GTBox(cls, np.array([x1, y1, x2, y2]), box.center.copy(), ann["instance_token"], ignore))
    return out


def lidar_in_image(nusc: NuScenes, cam_sd_token: str) -> tuple[np.ndarray, np.ndarray]:
    """Project the keyframe's LIDAR_TOP sweep into a camera image.

    Returns (uv, depth): pixel coordinates (2, N) and camera-frame depth (N,) in metres.
    """
    cam = nusc.get("sample_data", cam_sd_token)
    sample = nusc.get("sample", cam["sample_token"])
    pts, depth, _ = nusc.explorer.map_pointcloud_to_image(sample["data"]["LIDAR_TOP"], cam_sd_token)
    return pts[:2], depth


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)
