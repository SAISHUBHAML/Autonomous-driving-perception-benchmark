#!/bin/bash
# Download and extract nuScenes v1.0-mini (~4 GB) into datasets/nuscenes/.
# nuScenes is licensed CC BY-NC-SA 4.0 (non-commercial); see https://www.nuscenes.org/terms-of-use
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/datasets/nuscenes"
URL="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz"

if [ -d "$DEST/v1.0-mini" ]; then
    echo "nuScenes mini already present at $DEST"
    exit 0
fi
mkdir -p "$DEST"
echo "Downloading nuScenes v1.0-mini to $DEST ..."
curl -fL -C - -o "$DEST/v1.0-mini.tgz" "$URL"
tar -xzf "$DEST/v1.0-mini.tgz" -C "$DEST"
rm "$DEST/v1.0-mini.tgz"
echo "Done: $DEST"
