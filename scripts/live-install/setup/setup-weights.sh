#!/usr/bin/env bash
set -Eeuo pipefail

echo "[WEIGHTS] Installing huggingface_hub CLI if needed..."
python -m pip install "huggingface_hub==0.23.2"

echo "[WEIGHTS] Downloading LAM assets..."
mkdir -p tmp

if [ ! -d "model_zoo/human_parametric_models" ] || [ ! -d "thirdparties" ]; then
  huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp

  if [ -f "./tmp/LAM_assets.tar" ]; then
    tar -xf ./tmp/LAM_assets.tar
    rm ./tmp/LAM_assets.tar
  fi

  if [ -f "./tmp/thirdparty_models.tar" ]; then
    tar -xf ./tmp/thirdparty_models.tar
    rm ./tmp/thirdparty_models.tar
  fi
else
  echo "[WEIGHTS] LAM assets appear to exist, skipping assets extraction."
fi

echo "[WEIGHTS] Downloading LAM-20K checkpoint..."
mkdir -p ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/

huggingface-cli download \
  3DAIGC/LAM-20K \
  --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/

rm -rf ./tmp

echo "[WEIGHTS] Done."
