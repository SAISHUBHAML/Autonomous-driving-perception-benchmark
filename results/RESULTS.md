# Results — nuScenes v1.0-mini test scenes scene-0103, scene-0916

Generated 2026-11-18T19:44:54 by `scripts/evaluate.py` (commit 2dedb68), mps, imgsz 960.

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

## LiDAR segmentation (ground removal + DBSCAN, LiDAR only)

81 LIDAR_TOP keyframe sweeps, 2358 GT objects of the evaluated classes within 50 m with ≥ 5 LiDAR points. DBSCAN eps 0.3 m, min_samples 5 (chosen on training scenes).

| Clusters | Recall % | Precision on annotated objects % | Over-segmented % | BEV centre error median / mean (m) | Recall 0-20m | Recall 20-40m | Recall 40-50m |
|---|---|---|---|---|---|---|---|
| 8301 | 63.2 | 25.5 | 22.4 | 0.53 / 0.76 | 83.9 | 38.3 | 1.7 |

Recall by class: car 56.9 %, truck 63.0 %, bus 75.8 %, motorcycle 60.0 %, bicycle 56.5 %, pedestrian 76.1 %.

### Camera-labelled clusters (late camera–LiDAR fusion)

| Model | Labelled clusters | Recall % (same class) | Precision % (same class) | Recall car | Recall truck | Recall bus | Recall motorcycle | Recall bicycle | Recall pedestrian |
|---|---|---|---|---|---|---|---|---|---|
| yolov8n-coco (zero-shot) | 1424 | 49.9 | 81.2 | 49.2 | 44.6 | 72.7 | 15.6 | 21.7 | 58.6 |
| yolov8n fine-tuned | 1253 | 38.7 | 70.5 | 41.4 | 31.5 | 30.3 | 4.4 | 8.7 | 42.4 |

## 3D multi-object tracking in the global frame (EKF, all 6 cameras fused, 2 Hz keyframes)

Matched to GT when the BEV centre distance is ≤ 2 m (same class). Measurement noise fitted on scene-0655.

| Model | Configuration | MOTA % | IDF1 % | ID switches | FP | Misses | BEV centre error median / mean (m) | Velocity error median (m/s) |
|---|---|---|---|---|---|---|---|---|
| yolov8n-coco (zero-shot) | camera-only, no filter | -7.1 | 34.9 | 179 | 1175 | 1476 | 0.95 / 0.96 | – |
| yolov8n-coco (zero-shot) | camera-only, EKF | -12.4 | 34.3 | 158 | 1281 | 1530 | 0.94 / 0.98 | 0.78 |
| yolov8n-coco (zero-shot) | camera+LiDAR, no filter | 18.8 | 47.5 | 172 | 814 | 1160 | 0.55 / 0.67 | – |
| yolov8n-coco (zero-shot) | camera+LiDAR, EKF | 19.3 | 47.9 | 164 | 812 | 1157 | 0.61 / 0.71 | 0.52 |
| yolov8n fine-tuned | camera-only, no filter | -25.1 | 24.2 | 140 | 1270 | 1894 | 1.14 / 1.12 | – |
| yolov8n fine-tuned | camera-only, EKF | -20.5 | 26.9 | 115 | 1226 | 1843 | 1.04 / 1.04 | 0.85 |
| yolov8n fine-tuned | camera+LiDAR, no filter | 18.6 | 44.0 | 146 | 653 | 1351 | 0.55 / 0.67 | – |
| yolov8n fine-tuned | camera+LiDAR, EKF | 18.0 | 43.9 | 164 | 653 | 1349 | 0.64 / 0.73 | 0.56 |

## Latency (detector only, warm-up excluded)

| Model | Device | Median ms | p95 ms |
|---|---|---|---|
| yolov8n-coco (zero-shot) | mps | 14.6 | 15.6 |
| yolov8n fine-tuned | mps | 14.5 | 15.7 |
