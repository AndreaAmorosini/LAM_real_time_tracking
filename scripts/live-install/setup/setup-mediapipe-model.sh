#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p model_zoo/mediapipe

python - <<'PY'
import urllib.request

url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
out = "model_zoo/mediapipe/face_landmarker.task"

urllib.request.urlretrieve(url, out)
print("saved", out)
PY
