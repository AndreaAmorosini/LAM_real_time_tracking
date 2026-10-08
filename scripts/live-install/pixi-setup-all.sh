#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="logs/setup"
LOG_FILE="$LOG_DIR/setup-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

step() {
  local name="$1"
  shift

  echo
  echo "================================================================"
  echo "[SETUP] START: $name"
  echo "================================================================"
  echo "[SETUP] Command: $*"
  echo

  local start
  start=$(date +%s)

  if "$@"; then
    local end
    end=$(date +%s)
    echo
    echo "[SETUP] OK: $name (${end-start}s)"
  else
    local code=$?
    local end
    end=$(date +%s)
    echo
    echo "[SETUP] FAILED: $name (${end-start}s)"
    echo "[SETUP] Exit code: $code"
    echo "[SETUP] Full log: $LOG_FILE"
    exit "$code"
  fi
}

echo "[SETUP] Log file: $LOG_FILE"
echo "[SETUP] PWD=$PWD"
echo "[SETUP] Python=$(which python || true)"
echo "[SETUP] Pip=$(which pip || true)"
echo "[SETUP] Node=$(which node || true)"
echo "[SETUP] NPM=$(which npm || true)"
echo "[SETUP] CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "[SETUP] PIXI_ENV_PREFIX=${PIXI_ENV_PREFIX:-}"

step "Fix script permission" pixi run fix-scripts-permissions
step "Python base dependencies" pixi run setup-python-base
step "PyTorch CUDA 12.1" pixi run setup-torch-cu121
step "Live tracking dependencies" pixi run setup-live
step "LAM weights and assets" pixi run setup-weights
step "FaceBoxes extension" pixi run setup-faceboxes
step "CUDA extensions" pixi run setup-cuda-ext
step "MediaPipe model" pixi run setup-mediapipe-model
step "Blender 4.0.2 standalone" pixi run setup-blender
step "FBX SDK Python" pixi run setup-fbx-sdk
step "WebGL npm install" pixi run setup-webgl
step "WebGL production build" pixi run webgl-build
step "Studio production build" pixi run studio-build
step "Doctor checks" pixi run doctor

echo
echo "================================================================"
echo "[SETUP] ALL DONE"
echo "================================================================"
echo "[SETUP] Log file: $LOG_FILE"
