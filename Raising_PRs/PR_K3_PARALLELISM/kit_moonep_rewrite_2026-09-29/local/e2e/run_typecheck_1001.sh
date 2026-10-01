#!/bin/bash
# 10-01: the h100 MoonEP cell (fake MoonEP) with SPMD type checking on and AC off, 3 steps; the recipe prints
# "MOONEP_TYPECHECK entered" each time the checker is entered. Usage: run_typecheck_1001.sh <tree> <out dir>
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_moonep_rewrite_2026-09-29
F=$K/local/fake_moonep; W=$1; D=$2; V=/workspace/venv_0928/bin; rm -rf $D; mkdir -p $D/cache
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( cd $W && MOONEP_AC=typecheck PYTHONPATH=$F:$K:. TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
  CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1200 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d \
  --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module moonep_local --config moonep_cell \
  --training.steps 3 --metrics.log_freq 1 --debug.seed 42 --debug.deterministic --dump-folder $D/out > $D/run.log 2>&1 )
echo "rc=$? tree $(git -C $W rev-parse --short HEAD)" > $D/rc; rm -rf $D/out $D/cache
