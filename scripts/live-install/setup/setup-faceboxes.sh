#!/usr/bin/env bash
set -Eeuo pipefail

echo "[FACEBOXES] Building FaceBoxes extension..."

ROOT="${PIXI_PROJECT_ROOT:-$(pwd)}"
FACEBOXES_DIR="$ROOT/external/landmark_detection/FaceBoxesV2/utils"

if [ ! -d "$FACEBOXES_DIR" ]; then
  echo "[FACEBOXES] Missing directory: $FACEBOXES_DIR"
  exit 1
fi

cd "$FACEBOXES_DIR"

echo "[FACEBOXES] Working dir: $(pwd)"
echo "[FACEBOXES] Python: $(which python)"
python --version

python build.py build_ext --inplace

echo "[FACEBOXES] Build output:"
ls -lah

echo "[FACEBOXES] Done."
