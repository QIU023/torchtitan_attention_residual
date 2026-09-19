#!/bin/bash
# Stage 1, third attempt. Two fixes over the last one:
#   torch and torchvision pinned to the same nightly date, since 0907 has no
#   torchvision partner and the resolver then finds no solution;
#   STAGE1_DONE written only if torch actually imports, so the marker gates.
set -x
rm -rf /workspace/venv_k3
uv venv --python 3.12 /workspace/venv_k3 || { echo "VENV_FAILED"; exit 1; }
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH
uv pip install "torch==2.15.0.dev20260906+cu126" "torchvision==0.30.0.dev20260906+cu126" \
    --index-url https://download.pytorch.org/whl/nightly/cu126
echo "TORCH_RC=$?"
if python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'gpus', torch.cuda.device_count())"; then
    echo "STAGE1_DONE"
else
    echo "STAGE1_FAILED torch does not import"
fi
