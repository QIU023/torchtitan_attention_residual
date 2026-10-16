#!/bin/bash
# Model-level fp32 gradient harness on the patched spare tree: <gpus> <tag>
. /workspace/venv_1006/bin/activate
cd /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_cpmm_prefix
n=$(echo $1 | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/main_pins_1016:/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_cpmm_prefix TORCHINDUCTOR_CACHE_DIR=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/dyncp_fix/cache_h2/ic TRITON_CACHE_DIR=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/dyncp_fix/cache_h2/tc   timeout 1800 torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/dyncp_fix/harness_cp.py > /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/dyncp_fix/harness_$2.log 2>&1
echo "rc=$?" >> /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/dyncp_fix/harness_$2.log
