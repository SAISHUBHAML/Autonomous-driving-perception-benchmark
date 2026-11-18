"""Grid-search the DBSCAN parameters of adpb/lidar.py on the TRAINING scenes only.

    python scripts/tune_clustering.py

Picks the (eps, min_samples) with the best F2 score on the train + validation scenes and writes
the whole grid to results/clustering_tuning.json. F2 weights recall above precision: clusters are
object proposals, and a missed object cannot be recovered later, while a spurious cluster (a pole,
a bush) can be rejected by the camera (see camera-labelled clusters in adpb/lidar.py). The test scenes are not touched; copy the chosen
values into adpb/lidar.py (EPS, MIN_SAMPLES) before running scripts/evaluate.py.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adpb import lidar as L  # noqa: E402
from adpb import nusc as N  # noqa: E402

EPS = [0.3, 0.4, 0.5, 0.6, 0.8]
MIN_SAMPLES = [3, 4, 5, 8]


def main():
    s = N.splits()
    scenes = s["train"] + s["val"]
    grid = []
    for eps, ms in itertools.product(EPS, MIN_SAMPLES):
        r = L.evaluate_scene_clusters(scenes, eps=eps, min_samples=ms)
        row = {"eps": eps, "min_samples": ms, "recall": r["recall"],
               "precision": r["precision_on_annotated_objects"], "f1": r["f1"], "f2": r["f2"],
               "clusters_per_sweep": r["clusters"] / r["sweeps"]}
        grid.append(row)
        print(f"eps={eps:.1f} min_samples={ms:2d}  recall={row['recall']:.3f}  "
              f"precision={row['precision']:.3f}  f2={row['f2']:.3f}", flush=True)
    best = max(grid, key=lambda r: r["f2"])
    print(f"best on train scenes: eps={best['eps']} min_samples={best['min_samples']}")
    out = N.ROOT / "results" / "clustering_tuning.json"
    out.write_text(json.dumps({"scenes": scenes, "objective": "f2", "best": best, "grid": grid}, indent=2))


if __name__ == "__main__":
    main()
