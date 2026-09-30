#!/bin/bash
# 09-30 box (115.124.123.240, 4 x H100 SXM, driver 560.35.03 + compat 570 = CUDA 12.8): torch built from the same
# commit as the 09-29 box's venv_src (torch git version 68e0ae4967e1, the 09-28 nightly's source), CUDA 12.8,
# sm90 only, so the 4312 table's numbers can be reproduced on the same kernels.
set -x
M=~/mep; mkdir -p $M; cd $M
C=68e0ae4967e1a7e39d914179aea0026036cc1a23
if [ ! -d pt ]; then
  git init -q pt && git -C pt remote add origin https://github.com/pytorch/pytorch.git
  git -C pt fetch -q --depth 1 origin $C && git -C pt checkout -q FETCH_HEAD
fi
cd pt; git log -1 --format="%H %s" > $M/pt_commit.txt
git submodule sync -q; git submodule update -q --init --recursive --depth 1 --jobs 48
[ -d $M/venv_src ] || uv venv $M/venv_src --python 3.12
. $M/venv_src/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-build.txt 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=176 \
  USE_CUDA=1 USE_ROCM=0 USE_XPU=0 BUILD_TEST=0 USE_CUSPARSELT=0 USE_CUDSS=0 USE_NVSHMEM=0 USE_MPI=0 \
  USE_SYSTEM_NCCL=0 USE_KINETO=1 USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1 CMAKE_BUILD_TYPE=Release
uv pip install build
time python -m build --wheel --no-isolation > $M/pt_build.log 2>&1
echo "build rc=$?"
ls -la dist/
echo BUILD_DONE
