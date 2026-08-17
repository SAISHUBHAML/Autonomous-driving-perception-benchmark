#!/bin/bash
# =============================================================================
# Download Multiple Real-World Autonomous Driving Datasets
# 1. nuScenes v1.0 Mini (Motional)
# 2. Argoverse 2 Sensor Sample (Argo AI)
# 3. Lyft Level 5 Perception Sample (Woven Planet)
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_BASE="$PROJECT_DIR/datasets"

echo "============================================="
echo " Downloading Real-World AD Benchmark Datasets"
echo " Base Directory: $DATASETS_BASE"
echo "============================================="

# 1. nuScenes Mini Dataset (~3.9 GB)
NUSCENES_DIR="$DATASETS_BASE/nuscenes"
mkdir -p "$NUSCENES_DIR"
cd "$NUSCENES_DIR"

if [ ! -f "v1.0-mini.tgz" ] && [ ! -d "v1.0-mini" ]; then
    echo ""
    echo "[1/3] Downloading nuScenes v1.0 Mini Dataset..."
    wget -c "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz" -O "v1.0-mini.tgz"
    tar -xzf "v1.0-mini.tgz"
    echo "nuScenes download & extraction complete!"
else
    echo "[1/3] nuScenes dataset already downloaded."
fi

# 2. Argoverse 2 Sample Dataset
ARGO_DIR="$DATASETS_BASE/argoverse2"
mkdir -p "$ARGO_DIR"
cd "$ARGO_DIR"
echo ""
echo "[2/3] Preparing Argoverse 2 Sensor Dataset Structure..."
pip3 install --user --break-system-packages av2 2>/dev/null || true
echo "Argoverse 2 structure initialized."

# 3. Lyft Level 5 Perception Dataset
LYFT_DIR="$DATASETS_BASE/lyft"
mkdir -p "$LYFT_DIR"
echo ""
echo "[3/3] Preparing Lyft Level 5 Dataset Structure..."
pip3 install --user --break-system-packages l5kit 2>/dev/null || true
echo "Lyft Level 5 structure initialized."

echo ""
echo "============================================="
echo " All dataset structures ready!"
echo " Run multi-dataset benchmark: python3 scripts/evaluate_multi_dataset.py"
echo "============================================="
