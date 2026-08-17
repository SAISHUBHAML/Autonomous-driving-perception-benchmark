# 📋 Development Roadmap & Tasks Checklist

This document tracks all completed development tasks, milestones, and future roadmap items for the **Autonomous Driving Perception & Tracking Stack**.

---

## 🎯 Phase 1: Environment & Infrastructure
- [x] Environment initialization & CUDA GPU setup (`scripts/setup_environment.sh`)
- [x] ROS2 Jazzy middleware workspace creation (`carla_ad_ws/`)
- [x] CARLA Simulator Docker orchestration configuration (`docker/docker-compose.yml`)
- [x] Portable path resolution implementation across all Python scripts (`os.path` anchoring)

---

## 🌐 Phase 2: Dataset Integration Pipeline
- [x] Automated downloading script for nuScenes v1.0 Mini dataset (`scripts/download_nuscenes.sh`)
- [x] Automated downloading script for KITTI, Waymo Open, & Argoverse 2 datasets (`scripts/download_large_datasets.sh`)
- [x] Dataset ingestion & keyframe discovery pipeline (14,012 keyframes)
- [x] Configurable 70% Train / 15% Val / 15% Test split partitioner

---

## 🚗 Phase 3: Perception & Sensor Fusion Engine
- [x] YOLOv8n object detection integration (`scripts/run_real_model_inference.py`)
- [x] HSV color space traffic light state classifier
- [x] LiDAR point cloud ground plane extraction using RANSAC
- [x] 3D obstacle clustering via DBSCAN spatial clustering
- [x] Extended Kalman Filter (EKF) state estimation for 3D localization

---

## 🎯 Phase 4: Tracking & Trajectory Management
- [x] ByteTrack multi-object tracker integration with low-confidence matching
- [x] Per-track EKF motion prediction for velocity & trajectory estimation
- [x] Zero-ID-switch track persistence across occlusions

---

## 📊 Phase 5: Evaluation & Dynamic Benchmarking
- [x] Full dataset evaluation runner with configurable epochs (`--epochs 10, 20, 50`)
- [x] Evaluation reporting pipeline exporting structured `full_dataset_report.json`
- [x] Dynamic epoch-aware loss decay & mAP convergence plotting (`scripts/evaluate_multi_dataset.py`)
- [x] Cross-dataset comparison matrix generator (`multi_dataset_cross_benchmark.png`)

---

## 🚀 Phase 6: Code Quality & GitHub Release
- [x] Repository cleanup & `.gitignore` specification
- [x] Git author identity configuration (`SAISHUBHAML`)
- [x] Public GitHub repository publication & SSH key authentication
- [x] Technical Documentation (`README.md`, `IMPLEMENTATION_PLAN.md`, `DEVELOPMENT_TASKS.md`)

---

## 🔮 Future Roadmap (Phase 7)
- [ ] Add TensorRT GPU optimization for sub-5ms inference
- [ ] Integrate Bird's-Eye-View (BEV) map generation
- [ ] Add HD map lane line constraint fusion
