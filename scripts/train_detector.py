"""Fine-tune YOLOv8 on nuScenes-mini keyframes (scene-level split, no test scenes used).

    python scripts/train_detector.py --epochs 30

Exports the train/val scenes from adpb.nusc.splits() to YOLO format (images are symlinked,
labels come from the projected 3D annotations), then trains with Ultralytics. The best
checkpoint is copied to weights/yolov8n_nuscenes.pt.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO  # noqa: E402

from adpb import nusc as N  # noqa: E402
from adpb.detection import default_device  # noqa: E402

OUT = N.ROOT / "datasets" / "yolo_nuscenes"


def export(split: str, scenes: list[str]) -> int:
    nusc = N.load()
    img_dir, lbl_dir = OUT / "images" / split, OUT / "labels" / split
    for d in (img_dir, lbl_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    tokens = N.keyframe_camera_tokens(nusc, scenes)
    for tok in tokens:
        sd = nusc.get("sample_data", tok)
        w, h = sd["width"], sd["height"]
        src = Path(N.image_path(nusc, tok))
        (img_dir / src.name).symlink_to(src)
        lines = []
        for g in N.gt_boxes(nusc, tok):
            if g.ignore:
                continue
            x1, y1, x2, y2 = g.xyxy
            lines.append(f"{N.CLASSES.index(g.cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (lbl_dir / (src.stem + ".txt")).write_text("\n".join(lines))
    return len(tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", default="weights/yolov8n.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    s = N.splits()
    print(f"train scenes {s['train']}: {export('train', s['train'])} images")
    print(f"val scenes   {s['val']}: {export('val', s['val'])} images")
    print(f"test scenes  {s['test']}: not used for training")
    data_yaml = OUT / "data.yaml"
    data_yaml.write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\n"
                         f"names: {dict(enumerate(N.CLASSES))}\n")

    model = YOLO(str(N.ROOT / args.base))
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device or default_device(), seed=0, deterministic=True,
                project=str(N.ROOT / "runs"), name="finetune", exist_ok=True, plots=True)
    best = N.ROOT / "runs" / "finetune" / "weights" / "best.pt"
    shutil.copy(best, N.ROOT / "weights" / "yolov8n_nuscenes.pt")
    print(f"saved {N.ROOT / 'weights' / 'yolov8n_nuscenes.pt'}")


if __name__ == "__main__":
    main()
