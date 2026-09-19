#!/bin/bash
# Stage 2: the tree's requirements, Attention Gym, cutlass-dsl and the MoonEP build.
# Follows the H200 recipe, with CUDA_HOME on the 12.6 toolkit stage 1 installed.
set -x
until grep -q STAGE1_DONE /workspace/stage1.log; do sleep 20; done
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH

uv pip install "grain==0.2.18" "datasets>=3.6.0,<4.8.0" tensorboard wandb "tyro>=1.0.5" \
    "tokenizers>=0.15.0" "renderers==0.1.11" safetensors einops pillow "spmd_types==0.2.5" av \
    "torch_remat @ git+https://github.com/meta-pytorch/remat.git@d302699b1c58f83fa2c7b03bc2593967e9530335" \
    ninja pytest expecttest "apache-tvm-ffi>=0.1.12"
echo "DEPS_RC=$?"

SETUPTOOLS_SCM_PRETEND_VERSION=0.0.8 uv pip install --no-deps -e /workspace/src/attn_gym
echo "AG_RC=$?"

# Attention Gym's KDA CuTe backend needs >= 4.5; PyPI has no 4.5.x, so 4.6.0.
uv pip install "nvidia-cutlass-dsl==4.6.0" "nvidia-cutlass-dsl-libs-cu12"
echo "CUTLASS_RC=$?"

python -c "import torch, attn_gym, spmd_types, torch_remat, grain, tyro; print('IMPORTS_OK', torch.__version__)"

cd /workspace/src/moonep
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=48
time uv pip install --no-build-isolation --no-deps -e .
echo "MOONEP_BUILD_RC=$?"

# MoonEP pins cutlass-dsl 4.4.2; on 4.6.0 its grad-reduce kernel needs one rename.
git apply /workspace/kit/moonep_onbox/h200/moonep_grad_reduce_cutlass46.patch && echo "CUTLASS46_PATCH_OK"
cd / && python -c "import moonep; from moonep import Buffer; print('MOONEP_OK', moonep.__file__)"
echo "MOONEP_IMPORT_RC=$?"
echo "STAGE2_DONE"
