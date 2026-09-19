#!/bin/bash
# Stage 1: CUDA 12.6 toolkit (toolkit only, never the cuda metapackage which pulls drivers)
# plus the venv and the torch nightly whose CUDA matches it, per the H200 recipe.
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cuda-toolkit-12-6
echo "NVCC126_RC=$?"
ls -d /usr/local/cuda-12.6 && /usr/local/cuda-12.6/bin/nvcc --version | tail -2

uv venv --python 3.12 /workspace/venv_k3
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH
uv pip install "torch==2.15.0.dev20260907+cu126" "torchvision" \
    --index-url https://download.pytorch.org/whl/nightly/cu126
echo "TORCH_RC=$?"
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'gpus', torch.cuda.device_count())"
echo "STAGE1_DONE"
