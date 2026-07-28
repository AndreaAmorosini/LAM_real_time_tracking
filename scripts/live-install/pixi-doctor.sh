#!/usr/bin/env bash
set -Eeuo pipefail

echo "=== Environment ==="
echo "PWD=$PWD"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "PIXI_ENV_PREFIX=${PIXI_ENV_PREFIX:-}"
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "LAM_BLENDER_PATH=${LAM_BLENDER_PATH:-}"

echo
echo "=== Binaries ==="
which python || true
which pip || true
which node || true
which npm || true
which nvcc || true

echo
echo "=== Python packages ==="
python - <<'PY'
import importlib

packages = [
    "torch",
    "cv2",
    "mediapipe",
    "fastapi",
    "uvicorn",
    "numpy",
    "transformers",
]

for name in packages:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(f"{name}: FAIL {repr(e)}")
        raise
PY

echo
echo "=== Torch CUDA ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "=== Blender ==="
BLENDER="${LAM_BLENDER_PATH:-thirdparties/blender/blender-4.0.2-linux-x64/blender}"

if [ ! -x "$BLENDER" ]; then
  echo "Blender missing or not executable: $BLENDER"
  exit 1
fi

"$BLENDER" --version

echo
echo "=== WebGL frontend ==="
test -d webgl_frontend/node_modules
test -d webgl_frontend/dist

echo
echo "=== Required files ==="
test -f model_zoo/mediapipe/face_landmarker.task
test -f configs/inference/lam-20k-8gpu.yaml
test -d model_zoo/lam_models/releases/lam/lam-20k/step_045500
test -x thirdparties/blender/blender-4.0.2-linux-x64/blender
test -d webgl_frontend/node_modules
test -d webgl_frontend/dist


echo
echo "Doctor OK"
