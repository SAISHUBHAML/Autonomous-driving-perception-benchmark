"""Evaluate detectors on the nuScenes-mini test scenes and write results/ (JSON, Markdown, figure).

    python scripts/evaluate.py                                   # pretrained + fine-tuned (if present)
    python scripts/evaluate.py --models pretrained=weights/yolov8n.pt

Every number written here is computed from model outputs and nuScenes annotations in this run.
"""

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import ultralytics  # noqa: E402

from adpb import nusc as N  # noqa: E402
from adpb.detection import Detector, default_device, evaluate_detection  # noqa: E402
from adpb.localization import RANGES, evaluate_localization  # noqa: E402
from adpb.tracking import evaluate_tracking  # noqa: E402

RESULTS = N.ROOT / "results"
TRACKERS = ["bytetrack.yaml", "botsort.yaml"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=N.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def evaluate_model(name: str, weights: str, imgsz: int, skip_tracking: bool) -> dict:
    nusc = N.load()
    test = N.splits()["test"]
    tokens = N.keyframe_camera_tokens(nusc, test)
    det = Detector(weights, imgsz=imgsz)
    print(f"[{name}] detection on {len(tokens)} keyframe images ...", flush=True)
    out = {"weights": weights, "detection": evaluate_detection(det, tokens)}
    print(f"[{name}] 3D localization ...", flush=True)
    out["localization"] = evaluate_localization(det, tokens)
    out["latency"] = det.benchmark_latency([N.image_path(nusc, t) for t in tokens])
    if not skip_tracking:
        out["tracking"] = {}
        for trk in TRACKERS:
            print(f"[{name}] tracking with {trk} ...", flush=True)
            out["tracking"][trk.removesuffix(".yaml")] = evaluate_tracking(weights, test, trk, imgsz)
    return out


def fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def write_markdown(res: dict) -> str:
    models = res["models"]
    L = [f"# Results — nuScenes v1.0-mini test scenes {', '.join(res['split']['test'])}", "",
         f"Generated {res['generated']} by `scripts/evaluate.py` (commit {res['git_commit']}), "
         f"{res['environment']['device']}, imgsz {res['imgsz']}.", "",
         "## 2D detection (COCO-style AP, %)", "",
         "| Model | mAP@50 | mAP@50:95 | " + " | ".join(f"AP50 {c}" for c in N.CLASSES) + " |",
         "|---|---|---|" + "---|" * len(N.CLASSES)]
    for m, r in models.items():
        d = r["detection"]
        L.append(f"| {m} | {fmt(d['mAP50'], 1)} | {fmt(d['mAP50_95'], 1)} | "
                 + " | ".join(fmt(d["per_class"][c]["AP50"], 1) for c in N.CLASSES) + " |")
    any_d = next(iter(models.values()))["detection"]
    L += ["", f"{any_d['images']} images, {any_d['gt_boxes']} GT boxes "
              f"({any_d['ignored_gt_boxes']} low-visibility boxes treated as ignore regions).", "",
          "## 3D center error of matched detections (m)", "",
          "| Model | Matched | With LiDAR depth | Camera-only mean / median | Camera+LiDAR mean / median |",
          "|---|---|---|---|---|"]
    for m, r in models.items():
        lo = r["localization"]
        c, f = lo["center_error_m"]["camera"], lo["center_error_m"]["fused"]
        L.append(f"| {m} | {lo['matched_objects']} | {lo['objects_with_lidar_depth']} | "
                 f"{fmt(c['mean'])} / {fmt(c['median'])} | {fmt(f['mean'])} / {fmt(f['median'])} |")
    L += ["", "By ground-truth range (Camera+LiDAR median, m):", "",
          "| Model | " + " | ".join(next(iter(models.values()))["localization"]["by_range"]) + " |",
          "|---|" + "---|" * len(RANGES)]
    for m, r in models.items():
        br = r["localization"]["by_range"]
        L.append(f"| {m} | " + " | ".join(f"{fmt(v['fused'].get('median'))} (n={v['fused']['n']})"
                                          for v in br.values()) + " |")
    if any("tracking" in r for r in models.values()):
        L += ["", "## 2D multi-object tracking (12 sequences = 2 scenes × 6 cameras)", "",
              "| Model | Tracker | MOTA % | IDF1 % | ID switches | FP | Misses | GT tracks | Mostly tracked |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, r in models.items():
            for t, x in r.get("tracking", {}).items():
                L.append(f"| {m} | {t} | {fmt(x['MOTA'], 1)} | {fmt(x['IDF1'], 1)} | {x['id_switches']} | "
                         f"{x['false_positives']} | {x['misses']} | {x['gt_tracks']} | {x['mostly_tracked']} |")
    L += ["", "## Latency (detector only, warm-up excluded)", "", "| Model | Device | Median ms | p95 ms |",
          "|---|---|---|---|"]
    for m, r in models.items():
        la = r["latency"]
        L.append(f"| {m} | {la['device']} | {la['median_ms']:.1f} | {la['p95_ms']:.1f} |")
    return "\n".join(L) + "\n"


def plot(res: dict, path: Path):
    models = res["models"]
    names = list(models)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    x = range(len(N.CLASSES))
    w = 0.8 / len(names)
    for k, m in enumerate(names):
        ap = [100 * (models[m]["detection"]["per_class"][c]["AP50"] or 0) for c in N.CLASSES]
        axs[0].bar([i + k * w for i in x], ap, w, label=m)
    axs[0].set_xticks([i + w * (len(names) - 1) / 2 for i in x], N.CLASSES, rotation=20)
    axs[0].set_ylabel("AP@50 (%)")
    axs[0].set_title("Detection AP@50 per class")
    axs[0].legend()

    ranges = list(models[names[0]]["localization"]["by_range"])
    for k, m in enumerate(names):
        br = models[m]["localization"]["by_range"]
        axs[1].plot(ranges, [br[r]["camera"].get("median") for r in ranges], "--o", label=f"{m}: camera")
        axs[1].plot(ranges, [br[r]["fused"].get("median") for r in ranges], "-s", label=f"{m}: camera+LiDAR")
    axs[1].set_ylabel("median 3D center error (m)")
    axs[1].set_xlabel("ground-truth range")
    axs[1].set_title("3D localization error")
    axs[1].legend(fontsize=8)

    labels, mota, idf1 = [], [], []
    for m in names:
        for t, r in models[m].get("tracking", {}).items():
            labels.append(f"{m.replace(' (', chr(10) + '(')}\n{t}")
            mota.append(100 * r["MOTA"])
            idf1.append(100 * r["IDF1"])
    if labels:
        xi = range(len(labels))
        axs[2].bar([i - 0.2 for i in xi], mota, 0.4, label="MOTA")
        axs[2].bar([i + 0.2 for i in xi], idf1, 0.4, label="IDF1")
        axs[2].axhline(0, color="k", lw=0.8)
        axs[2].set_xticks(list(xi), labels, fontsize=8)
        axs[2].set_ylabel("%")
        axs[2].set_title("Tracking")
        axs[2].legend()
    fig.suptitle(f"nuScenes v1.0-mini, test scenes {', '.join(res['split']['test'])}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="name=weights pairs")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--skip-tracking", action="store_true")
    args = ap.parse_args()

    if args.models:
        models = dict(m.split("=", 1) for m in args.models)
    else:
        models = {"yolov8n-coco (zero-shot)": "weights/yolov8n.pt"}
        if (N.ROOT / "weights" / "yolov8n_nuscenes.pt").exists():
            models["yolov8n fine-tuned"] = "weights/yolov8n_nuscenes.pt"

    res = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset": N.VERSION,
        "split": N.splits(),
        "classes": N.CLASSES,
        "imgsz": args.imgsz,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "ultralytics": ultralytics.__version__, "platform": platform.platform(),
                        "device": default_device()},
        "models": {name: evaluate_model(name, str(N.ROOT / w), args.imgsz, args.skip_tracking)
                   for name, w in models.items()},
    }
    for r in res["models"].values():
        r["weights"] = str(Path(r["weights"]).relative_to(N.ROOT))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(res, indent=2))
    (RESULTS / "RESULTS.md").write_text(write_markdown(res))
    plot(res, RESULTS / "summary.png")
    print((RESULTS / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
