#!/bin/bash
. /workspace/venv_0928/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
for c in standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp; do
  ( cd /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_moonep_rebuild && FAKE_MULTICAST=1 PROBE_STEPS=2 PROBE_DET=1 PYTHONPATH=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/moonep_hook/mc_shim:/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_moonep_perf_2026-10-03:/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/fake_moonep_async:/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_moonep_rebuild TORCHINDUCTOR_CACHE_DIR=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/moonep_hook/smoke_layouts/ic TRITON_CACHE_DIR=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/moonep_hook/smoke_layouts/tc \
    CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 LOG_RANK=0,1,2,3 MODULE=moonep_probe_1003 CONFIG=${c}_cell timeout 900 ./run_train.sh --output-dir /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/moonep_hook/smoke_layouts/out_$c > /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/moonep_hook/smoke_layouts/$c.log 2>&1 )
  echo "$c rc=$? $(sed 's/\x1b\[[0-9;]*m//g' /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/moonep_hook/smoke_layouts/$c.log | grep -a -o 'step: *2 .*grad_norm: *[0-9.]*' | head -1)" >> /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/moonep_hook/smoke_layouts/progress.txt
done
echo done >> /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/moonep_hook/smoke_layouts/progress.txt
