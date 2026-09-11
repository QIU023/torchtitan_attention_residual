#!/bin/bash
set -uo pipefail
exec > /workspace/setup.log 2>&1
echo "=== $(date -Iseconds) setup start"
source /venv/main/bin/activate
echo "--- torch nightly cu130"
uv pip install --index-url https://download.pytorch.org/whl/nightly/cu130 --pre \
  "torch==2.15.0.dev20260906+cu130" "torchvision==0.30.0.dev20260906+cu130" 2>&1 | tail -5
python -c "import torch;print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
echo "--- deps"
uv pip install "spmd_types==0.2.5" "torchao==0.18.0" "torchdata==0.11.0" "grain==0.2.18" \
  "datasets>=3.6.0,<4.8.0" tensorboard wandb "tyro>=1.0.5" "tokenizers>=0.15.0" safetensors einops pillow \
  "attn-gym[linear]==0.0.8" "nvidia-cutlass-dsl==4.6.0" "cuda-bindings==13.3.1" 2>&1 | tail -5
uv pip install "torch_remat @ git+https://github.com/meta-pytorch/remat.git@d302699b1c58f83fa2c7b03bc2593967e9530335" 2>&1 | tail -3
echo "--- attn-gym latest main"
rm -rf /workspace/attn_gym_up
git clone -q https://github.com/meta-pytorch/attention-gym.git /workspace/attn_gym_up && \
  git -C /workspace/attn_gym_up log --oneline -1
echo "--- torchtitan fork"
rm -rf /workspace/titan
git clone -q https://github.com/QIU023/torchtitan.git /workspace/titan && cd /workspace/titan && \
  git fetch -q origin pp_review4 && git checkout -q 8aea9ef03 && git log --oneline -1 && git rev-parse HEAD
echo "=== $(date -Iseconds) setup done"
touch /workspace/setup.done
