#!/bin/bash
# The box's driver (560.35.05, forward compat to 12.8) cannot run cu130 wheels (error 803 with the
# 13.0 compat libs) and the last cu126 nightly (09-07) predates main's per-edge PP P2P (#4540) and
# unshard lookahead (#4592); so torch is built here from the nightly branch against CUDA 12.8, sm90.
set -x
M=~/mep; cd $M
[ -d pt ] || git clone -q --depth 1 -b nightly https://github.com/pytorch/pytorch.git pt
cd pt; git log -1 --format="%H %s" > $M/pt_commit.txt
git submodule sync -q; git submodule update -q --init --recursive --depth 1 --jobs 32
[ -d $M/venv_src ] || uv venv $M/venv_src --python 3.12
. $M/venv_src/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-build.txt 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=112 \
  USE_CUDA=1 USE_ROCM=0 USE_XPU=0 BUILD_TEST=0 USE_CUSPARSELT=0 USE_CUDSS=0 USE_NVSHMEM=0 USE_MPI=0 \
  USE_SYSTEM_NCCL=0 USE_KINETO=1 USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1 CMAKE_BUILD_TYPE=Release
uv pip install build
time python -m build --wheel --no-isolation > $M/pt_build.log 2>&1
ls -la dist/
echo BUILD_DONE
