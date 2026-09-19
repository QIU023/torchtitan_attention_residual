#!/bin/bash
# Retry of stage 1 with timeouts, after the first attempt hung with a dead mirror connection.
set -x
export DEBIAN_FRONTEND=noninteractive
APTOPT="-o Acquire::http::Timeout=25 -o Acquire::https::Timeout=25 -o Acquire::Retries=4"
apt-get $APTOPT install -y cuda-toolkit-12-6
echo "NVCC126_RC=$?"
/usr/local/cuda-12.6/bin/nvcc --version | tail -2

uv venv --python 3.12 /workspace/venv_k3
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH
uv pip install "torch==2.15.0.dev20260907+cu126" "torchvision" \
    --index-url https://download.pytorch.org/whl/nightly/cu126
echo "TORCH_RC=$?"
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'gpus', torch.cuda.device_count())"
echo "STAGE1_DONE"
