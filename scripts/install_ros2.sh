#!/bin/bash
# =============================================================================
# ROS2 Jazzy Installation Script for Ubuntu 24.04 (Noble)
# CARLA-ROS2 Autonomous Driving Stack
# =============================================================================

set -e

echo "============================================="
echo " ROS2 Jazzy Installation for Ubuntu 24.04"
echo "============================================="

# ---- 1. Set locale ----
echo "[1/7] Setting up locale..."
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ---- 2. Add ROS2 apt repository ----
echo "[2/7] Adding ROS2 repository..."
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ---- 3. Install ROS2 Jazzy Desktop ----
echo "[3/7] Installing ROS2 Jazzy Desktop (this may take a while)..."
sudo apt update
sudo apt install -y ros-jazzy-desktop

# ---- 4. Install development tools ----
echo "[4/7] Installing development tools..."
sudo apt install -y \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-pip

# ---- 5. Install additional ROS2 packages ----
echo "[5/7] Installing additional ROS2 packages..."
sudo apt install -y \
    ros-jazzy-vision-msgs \
    ros-jazzy-diagnostic-msgs \
    ros-jazzy-rviz2 \
    ros-jazzy-rqt \
    ros-jazzy-rqt-graph \
    ros-jazzy-rqt-topic \
    ros-jazzy-rqt-console \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-pcl-conversions \
    ros-jazzy-pcl-ros

# ---- 6. Initialize rosdep ----
echo "[6/7] Initializing rosdep..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

# ---- 7. Add ROS2 to .bashrc ----
echo "[7/7] Configuring shell environment..."
BASHRC_LINE="source /opt/ros/jazzy/setup.bash"
if ! grep -qF "$BASHRC_LINE" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# ROS2 Jazzy" >> ~/.bashrc
    echo "$BASHRC_LINE" >> ~/.bashrc
    echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' >> ~/.bashrc
    echo "Added ROS2 Jazzy to ~/.bashrc"
else
    echo "ROS2 already in ~/.bashrc"
fi

echo ""
echo "============================================="
echo " ROS2 Jazzy installation complete!"
echo " Run: source ~/.bashrc"
echo " Verify: ros2 --version"
echo "============================================="
