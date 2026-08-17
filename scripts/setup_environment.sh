#!/bin/bash
# =============================================================================
# Full Environment Setup Script
# CARLA-ROS2 Autonomous Driving Stack
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$PROJECT_DIR/carla_ad_ws"

echo "============================================="
echo " CARLA-ROS2 Full Environment Setup"
echo "============================================="
echo "Project directory: $PROJECT_DIR"

# ---- 1. Install ROS2 Jazzy ----
echo ""
echo "[1/6] Installing ROS2 Jazzy..."
bash "$SCRIPT_DIR/install_ros2.sh"
source /opt/ros/jazzy/setup.bash

# ---- 2. Install Python dependencies ----
echo ""
echo "[2/6] Installing Python dependencies..."
pip3 install --user --upgrade pip --break-system-packages || true

pip3 install --user --break-system-packages \
    numpy>=1.24 \
    scipy>=1.11 \
    opencv-python>=4.8 \
    opencv-contrib-python>=4.8 \
    Pillow>=10.0 \
    scikit-learn>=1.3 \
    matplotlib>=3.7 \
    transforms3d>=0.4 \
    open3d>=0.17

# PyTorch with CUDA support
echo "Installing PyTorch..."
pip3 install --user --break-system-packages torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Ultralytics (YOLOv8)
echo "Installing Ultralytics (YOLOv8)..."
pip3 install --user --break-system-packages ultralytics>=8.0

# CARLA Python API
echo "Installing CARLA Python API..."
pip3 install --user --break-system-packages "carla>=0.9.15"

# ---- 3. Install Docker & NVIDIA Container Toolkit ----
echo ""
echo "[3/6] Setting up Docker for CARLA..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "NOTE: Log out and back in for Docker group permissions to take effect."
else
    echo "Docker already installed."
fi

# NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
else
    echo "NVIDIA Container Toolkit already installed."
fi

# ---- 4. Pull CARLA Docker image ----
echo ""
echo "[4/6] Pulling CARLA Docker image..."
docker pull carlasim/carla:0.9.15

# ---- 5. Download YOLOv8 weights ----
echo ""
echo "[5/6] Downloading YOLOv8 model weights..."
MODELS_DIR="$WORKSPACE_DIR/src/perception/models"
mkdir -p "$MODELS_DIR"
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('YOLOv8n weights downloaded successfully')
" 2>/dev/null && mv yolov8n.pt "$MODELS_DIR/" 2>/dev/null || echo "Will download weights on first run"

# ---- 6. Build ROS2 workspace ----
echo ""
echo "[6/6] Building ROS2 workspace..."
cd "$WORKSPACE_DIR"
if [ -d "src" ]; then
    source /opt/ros/jazzy/setup.bash
    rosdep install --from-paths src --ignore-src -r -y || true
    colcon build --symlink-install
    echo "Workspace built successfully!"

    # Add workspace to .bashrc
    WS_LINE="source $WORKSPACE_DIR/install/setup.bash"
    if ! grep -qF "$WS_LINE" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# CARLA-ROS2 Workspace" >> ~/.bashrc
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo "$WS_LINE" >> ~/.bashrc
    fi
else
    echo "Workspace source directory not found. Build after creating packages."
fi

echo ""
echo "============================================="
echo " Environment setup complete!"
echo ""
echo " Next steps:"
echo "   1. source ~/.bashrc"
echo "   2. bash scripts/launch_carla.sh"
echo "   3. ros2 launch carla_ad_launch full_stack.launch.py"
echo "============================================="
