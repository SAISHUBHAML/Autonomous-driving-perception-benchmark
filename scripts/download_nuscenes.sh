#!/bin/bash
# =============================================================================
# Download Official nuScenes v1.0 Mini Dataset (3.9 GB)
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$PROJECT_DIR/datasets/nuscenes"

echo "============================================="
echo " Downloading Official nuScenes v1.0 Mini Dataset"
echo " Target Directory: $DATASET_DIR"
echo "============================================="

# 1. Install nuScenes devkit
echo "[1/3] Installing nuscenes-devkit..."
pip3 install --user --break-system-packages nuscenes-devkit matplotlib opencv-python

# 2. Create directory and download dataset
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

TAR_FILE="v1.0-mini.tgz"
URL="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz"

if [ ! -f "$TAR_FILE" ] && [ ! -d "v1.0-mini" ]; then
    echo "[2/3] Downloading nuScenes v1.0-mini.tgz (~3.9 GB)..."
    wget -c "$URL" -O "$TAR_FILE"
else
    echo "[2/3] nuScenes archive or directory already exists."
fi

# 3. Extract dataset
if [ -f "$TAR_FILE" ] && [ ! -d "v1.0-mini" ]; then
    echo "[3/3] Extracting nuScenes dataset..."
    tar -xzf "$TAR_FILE"
    echo "Extraction complete!"
fi

echo ""
echo "============================================="
echo " nuScenes dataset ready at: $DATASET_DIR"
echo " Run benchmark: python3 scripts/evaluate_on_nuscenes.py"
echo "============================================="
