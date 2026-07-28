#!/usr/bin/env bash
set -Eeuo pipefail

python -m pip install \
  torch==2.3.0 \
  torchvision==0.18.0 \
  torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  -U xformers==0.0.26.post1 \
  --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
