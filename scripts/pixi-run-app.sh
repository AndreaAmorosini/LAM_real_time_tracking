#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIXI_PROJECT_ROOT:-$(pwd)}"
ENV_PREFIX="${CONDA_PREFIX:-${PIXI_ENV_PREFIX:-$ROOT/.pixi/envs/default}}"

export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

export CUDA_HOME="$ENV_PREFIX"
export CUDA_PATH="$ENV_PREFIX"

export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CPATH="$ENV_PREFIX/include:$ENV_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export C_INCLUDE_PATH="$ENV_PREFIX/include:$ENV_PREFIX/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$ENV_PREFIX/include:$ENV_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"

export NUMBA_THREADING_LAYER=omp
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="localhost,127.0.0.1,0.0.0.0"

# Metto la cache delle estensioni torch dentro il progetto,
# così è più facile pulirla e non dipende da ~/.cache.
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_extensions"

echo "ROOT=$ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "CPATH=$CPATH"
echo "LIBRARY_PATH=$LIBRARY_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CC=$CC"
echo "CXX=$CXX"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"

find "$ENV_PREFIX" -name cuda_runtime.h | head || true
find "$ENV_PREFIX" -name cuda_runtime_api.h | head || true

python app_lam.py
