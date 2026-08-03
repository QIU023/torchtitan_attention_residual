#!/bin/bash
# Bootstrap node 1 (same vast template as node 0 -> torch 2.12.0+cu130
# preinstalled in /venv/main). Reproduces SESSION_HANDOFF_2026-07-24 sec 2
# exactly, plus the submodule pins from the logbook.
set -eux

cd /workspace
if [ ! -d torchtitan_attention_residual ]; then
  git clone https://github.com/QIU023/torchtitan_attention_residual
fi
cd torchtitan_attention_residual
git pull --ff-only
# submodules are pinned by the logbook; torchtitan is the only one training needs
git submodule update --init torchtitan

source /venv/main/bin/activate
uv pip install fla-core==0.5.1 torchao==0.17.0 "transformers>=5" datasets \
  tiktoken blobfile safetensors pandas pyarrow pytest expecttest
uv pip install -r torchtitan/requirements.txt

cd torchtitan
python - <<'PY'
import torch, fla, torchao
print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
import torchtitan
from torchtitan.experiments.kimi_k3.config_registry import kimi_k3_debugmodel8h as f
print("flavor ok:", f().model_spec.flavor)
PY

# quick single-node sanity before any 2-node attempt (2 GPUs, 2 steps)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29777 -m torchtitan.train \
  --module kimi_k3 --config kimi_k3_debugmodel --checkpoint.no-enable \
  --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 2 \
  --parallelism.data_parallel_shard_degree 2 --dump-folder /workspace/out_worker_sanity 2>&1 \
  | grep -E "step: +[0-9]+ +loss" | head -4

echo "WORKER READY"
