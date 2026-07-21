#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIXI_PROJECT_ROOT:-$(pwd)}"
ENV_PREFIX="${CONDA_PREFIX:-${PIXI_ENV_PREFIX:-$ROOT/.pixi/envs/default}}"

export PATH="$ENV_PREFIX/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-$ENV_PREFIX}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$ENV_PREFIX/include:$ENV_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CPATH="$ENV_PREFIX/include:$ENV_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"

find_bin() {
  for bin in "$@"; do
    if command -v "$bin" >/dev/null 2>&1; then
      command -v "$bin"
      return 0
    fi
  done
  return 1
}

CC_BIN="$(find_bin x86_64-conda-linux-gnu-cc gcc)"
CXX_BIN="$(find_bin x86_64-conda-linux-gnu-c++ x86_64-conda-linux-gnu-g++ g++)"

export CC="$CC_BIN"
export CXX="$CXX_BIN"
export CUDAHOSTCXX="$CXX_BIN"
export CMAKE_C_COMPILER="$CC_BIN"
export CMAKE_CXX_COMPILER="$CXX_BIN"

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"

if [ -d "$ENV_PREFIX/include/cub" ] || [ -d "$ENV_PREFIX/targets/x86_64-linux/include/cub" ]; then
  export CUB_HOME="$ENV_PREFIX"
fi

echo "ROOT=$ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDAHOSTCXX=$CUDAHOSTCXX"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "MAX_JOBS=$MAX_JOBS"

which python
python -m pip --version
which nvcc || true
nvcc --version || true
"$CC" --version | head -n 1
"$CXX" --version | head -n 1

python -m pip install --no-build-isolation -v git+https://github.com/facebookresearch/pytorch3d.git
python -m pip install --no-build-isolation -v git+https://github.com/ashawkey/diff-gaussian-rasterization/
python -m pip install --no-build-isolation -v "nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling"
python -m pip install --no-build-isolation -v git+https://github.com/camenduru/simple-knn/
