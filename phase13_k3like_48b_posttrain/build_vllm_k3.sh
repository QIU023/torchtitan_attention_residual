#!/usr/bin/env bash
# Source build of vLLM with the K3 branch's kernels compiled.
# Deliberately NOT VLLM_USE_PRECOMPILED: that reuses a release wheel's binaries,
# which have no _flashkda_C and no _C::situ_and_mul.
set -uo pipefail
cd /workspace/vllm_k3
unset VLLM_USE_PRECOMPILED
export TORCH_CUDA_ARCH_LIST="12.0"     # sm_120, the only GPU on this box
export MAX_JOBS=32
export NVCC_THREADS=4
export CMAKE_BUILD_TYPE=Release
export VIRTUAL_ENV=/venv/vllm_k3
export PATH="$HOME/.cargo/bin:$PATH"
echo "=== start $(date -u +%H:%M:%S) arch=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS ==="
uv pip install --python /venv/vllm_k3/bin/python --no-build-isolation -v -e . 2>&1
echo "=== exit=$? $(date -u +%H:%M:%S) ==="
