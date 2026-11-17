# Results — nuScenes v1.0-mini test scenes scene-0103, scene-0916

Generated 2026-11-17T15:49:32 by `scripts/evaluate.py` (commit db188d5-dirty), mps, imgsz 960.

## 2D detection (COCO-style AP, %)

| Model | mAP@50 | mAP@50:95 | AP50 car | AP50 truck | AP50 bus | AP50 motorcycle | AP50 bicycle | AP50 pedestrian |
|---|---|---|---|---|---|---|---|---|
| yolov8n-coco (zero-shot) | 41.4 | 16.6 | 68.3 | 67.2 | 51.2 | 16.0 | 8.7 | 36.7 |
| yolov8n fine-tuned | 36.0 | 18.7 | 75.9 | 50.2 | 7.6 | 12.3 | 8.4 | 61.6 |

486 images, 3460 GT boxes (1923 low-visibility boxes treated as ignore regions).

## 3D center error of matched detections (m)

| Model | Matched | With LiDAR depth | Camera-only mean / median | Camera+LiDAR mean / median |
|---|---|---|---|---|
| yolov8n-coco (zero-shot) | 1906 | 1543 | 2.15 / 1.38 | 1.68 / 0.68 |
| yolov8n fine-tuned | 1955 | 1748 | 2.89 / 2.25 | 1.82 / 0.69 |

By ground-truth range (Camera+LiDAR median, m):

| Model | 0-20m | 20-40m | >40m |
|---|---|---|---|
| yolov8n-coco (zero-shot) | 0.54 (n=1167) | 0.87 (n=626) | 2.64 (n=113) |
| yolov8n fine-tuned | 0.55 (n=1235) | 0.89 (n=640) | 4.77 (n=80) |

## 2D multi-object tracking (12 sequences = 2 scenes × 6 cameras)

| Model | Tracker | MOTA % | IDF1 % | ID switches | FP | Misses | GT tracks | Mostly tracked |
|---|---|---|---|---|---|---|---|---|
| yolov8n-coco (zero-shot) | bytetrack | 4.0 | 39.3 | 194 | 1084 | 2042 | 570 | 86 |
| yolov8n-coco (zero-shot) | botsort | 16.5 | 48.8 | 171 | 1011 | 1706 | 570 | 154 |
| yolov8n fine-tuned | bytetrack | 26.8 | 47.5 | 220 | 439 | 1872 | 570 | 109 |
| yolov8n fine-tuned | botsort | 37.3 | 56.4 | 160 | 357 | 1651 | 570 | 172 |

## Latency (detector only, warm-up excluded)

| Model | Device | Median ms | p95 ms |
|---|---|---|---|
| yolov8n-coco (zero-shot) | mps | 14.5 | 15.6 |
| yolov8n fine-tuned | mps | 14.5 | 15.8 |
