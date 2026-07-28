#!/usr/bin/env bash
set -Eeuo pipefail

WHEEL="/tmp/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl"
URL="https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl"

if [ ! -f "$WHEEL" ]; then
  echo "[FBX] Downloading FBX SDK wheel..."
  wget -O "$WHEEL" "$URL"
else
  echo "[FBX] Wheel already exists: $WHEEL"
fi

python -m pip install "$WHEEL"

python - <<'PY'
try:
    import fbx
    print("[FBX] import fbx OK")
except Exception as e:
    print("[FBX] import fbx FAILED:", repr(e))
    raise
PY
