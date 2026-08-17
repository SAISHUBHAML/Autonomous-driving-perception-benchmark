# CARLA-ROS2: Modular Autonomous Driving Stack

A complete autonomous driving system for the CARLA simulator, built with ROS2.
Implements a full **Perception → Planning → Control** pipeline using modular ROS2 nodes.

## 🎯 What It Does

The vehicle autonomously:
- **Follows lanes** using camera and LiDAR perception
- **Detects vehicles, pedestrians, and cyclists** with YOLOv8
- **Detects and obeys traffic lights** (Red → Stop, Green → Go)
- **Maintains safe following distance** (2-second rule)
- **Stops at red lights** with smooth deceleration
- **Avoids obstacles** in the lane
- **Changes lanes** to overtake slow vehicles
- **Reaches a destination** through urban traffic (Town03)

## 📄 Project Documentation & Specifications
- 📐 **[Implementation Plan & Architecture](docs/IMPLEMENTATION_PLAN.md)**: Full design specifications, sensor fusion algorithms, and dataset split strategy.
- 📋 **[Development Tasks & Roadmap](docs/DEVELOPMENT_TASKS.md)**: Task checklist, completed milestones, and future features roadmap.

## 🏗️ Architecture

```
                    CARLA Simulator (Docker)
                           │
              ┌────────────┴────────────┐
              │                         │
           Camera                     LiDAR
              │                         │
              ▼                         ▼
     ┌─────────────────┐     ┌──────────────────┐
     │ Perception Node │     │ LiDAR Processing │
     │   (YOLOv8)      │     │  (RANSAC+DBSCAN) │
     └────────┬────────┘     └────────┬─────────┘
              │                       │
              └───────────┬───────────┘
                          ▼
                ┌──────────────────┐
                │  Sensor Fusion   │
                │     (EKF)        │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Object Tracking  │
                │  (ByteTrack)     │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Decision Making  │
                │    (FSM)         │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │  Path Planning   │
                │ (Hybrid A*)      │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Vehicle Control  │
                │(PID+Pure Pursuit)│
                └────────┬─────────┘
                         ▼
                    CARLA Vehicle
```

## 🛠️ Technology Stack

| Technology | Purpose |
|-----------|---------|
| **Ubuntu 24.04** | Operating System |
| **ROS2 Jazzy** | Robot middleware |
| **CARLA 0.9.15** | Driving simulator (Docker) |
| **Python 3.12** | Primary language |
| **PyTorch + CUDA** | Deep learning inference |
| **YOLOv8 (Ultralytics)** | Object detection |
| **OpenCV** | Image processing |
| **NumPy / SciPy** | Numerical computation |
| **scikit-learn** | DBSCAN clustering |

## ROS2 Packages

| Package | Description | Key Algorithms |
|---------|-------------|---------------|
| `carla_ad_msgs` | Custom message definitions | 9 message types |
| `carla_bridge` | CARLA ↔ ROS2 interface | Sync mode, sensor spawning |
| `perception` | Camera object detection | YOLOv8n, HSV traffic light classification |
| `lidar_processing` | LiDAR point cloud pipeline | RANSAC ground removal, DBSCAN clustering |
| `sensor_fusion` | Multi-sensor fusion | Extended Kalman Filter, camera-LiDAR projection |
| `object_tracking` | Multi-object tracking | ByteTrack, per-track Kalman filters |
| `decision_making` | Behavioral decisions | Finite State Machine (5 states) |
| `path_planning` | Trajectory generation | Hybrid A*, cubic spline smoothing |
| `vehicle_control` | Throttle/steering | PID (longitudinal) + Pure Pursuit (lateral) |
| `carla_ad_launch` | System orchestration | Sequenced launch files |

## Quick Start

### Prerequisites
- Ubuntu 24.04
- NVIDIA GPU with CUDA support
- Docker with nvidia-container-toolkit

### 1. Setup Environment
```bash
cd Project_1
chmod +x scripts/*.sh
bash scripts/setup_environment.sh
source ~/.bashrc
```

### 2. Start CARLA Simulator
```bash
bash scripts/launch_carla.sh
```

### 3. Build Workspace
```bash
cd carla_ad_ws
colcon build --symlink-install
source install/setup.bash
```

### 4. Launch Full Stack
```bash
ros2 launch carla_ad_launch full_stack.launch.py
```

### 5. Monitor (in separate terminals)
```bash
# View all topics
ros2 topic list

# Monitor detections
ros2 topic echo /tracking/tracked_objects

# Monitor decisions
ros2 topic echo /planning/driving_command

# Check rates
ros2 topic hz /carla/camera/image
```

## ROS2 Topic Map

```
/carla/camera/image          ─→  Perception Node
/carla/lidar/points          ─→  LiDAR Processing
/carla/gnss                  ─→  Sensor Fusion
/carla/imu                   ─→  Sensor Fusion
/carla/ego/odometry          ─→  Planner, Controller
/carla/traffic_lights        ─→  Decision Maker
/perception/detections_2d    ─→  Sensor Fusion
/perception/lidar_objects    ─→  Sensor Fusion
/perception/fused_objects    ─→  Object Tracker
/tracking/tracked_objects    ─→  Decision Maker
/planning/driving_command    ─→  Path Planner
/planning/trajectory         ─→  Vehicle Controller
/control/vehicle_cmd         ─→  CARLA Bridge
```

## Decision Making States

| State | Trigger | Behavior |
|-------|---------|----------|
| `LANE_FOLLOWING` | Default | Follow lane at target speed |
| `STOP_AT_LIGHT` | Red/yellow traffic light | Smooth deceleration to stop |
| `LANE_CHANGE` | Slow vehicle ahead | Execute lane change maneuver |
| `OBSTACLE_AVOIDANCE` | Static obstacle | Generate avoidance trajectory |
| `EMERGENCY_STOP` | TTC < 2 seconds | Maximum braking |

## Project Structure

```
Project_1/
├── README.md
├── scripts/                    # Setup & launch scripts
├── docker/                     # CARLA Docker config
└── carla_ad_ws/               # ROS2 workspace
    └── src/
        ├── carla_ad_msgs/     # Custom messages
        ├── carla_bridge/      # CARLA interface
        ├── perception/        # YOLOv8 detection
        ├── lidar_processing/  # Point cloud processing
        ├── sensor_fusion/     # EKF fusion
        ├── object_tracking/   # ByteTrack tracker
        ├── decision_making/   # FSM decisions
        ├── path_planning/     # Hybrid A* planning
        ├── vehicle_control/   # PID + Pure Pursuit
        └── carla_ad_launch/   # Launch files
```

##  Configuration

All nodes are configurable via YAML parameters in each package's `config/` directory.
Key parameters can be adjusted at launch time.

## License

MIT License
