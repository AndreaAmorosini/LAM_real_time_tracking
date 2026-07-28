#!/usr/bin/env bash
set -Eeuo pipefail

python -m pip install "numpy==1.23.0"
python -m pip install "mediapipe==0.10.14"
python -m pip install "websockets"
python -m pip install \
  "gradio==3.44.3" \
  "fastapi==0.103.2" \
  "starlette==0.27.0" \
  "pydantic==1.10.15" \
  "uvicorn==0.23.2" \
  "anyio==3.7.1"

python -m pip install --no-deps --force-reinstall "transformers==4.41.2"
python -m pip install --no-deps --force-reinstall "tokenizers==0.19.1"
python -m pip install --no-deps --force-reinstall "huggingface_hub==0.23.2"
python -m pip install --force-reinstall "numpy==1.23.0"

python -m pip check || true
