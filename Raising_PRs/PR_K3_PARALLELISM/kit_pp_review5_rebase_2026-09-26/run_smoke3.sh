#!/bin/bash
# pp4 x vpp4 under the default DistMuon with every stage owning a transformer layer, 10 steps.
set -uo pipefail
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/smoke_pr5/runs
cd /tmp/wt_pp_review5
D=$R/pp4vpp4_muon_lps; rm -rf $D; mkdir -p $D; cp -r $R/pp4vpp4/cache $D/cache
PYTHONPATH=$S/smoke_pr5:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
/workspace/venv_bfx9/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  --role rank --tee 3 \
  -m torchtitan.train --module smoke_pr5 --config kimi_k3_pp4_vpp4_default_optimizer_layer_per_stage --training.steps 10 --dump-folder $D/out > $D/run.log 2>&1
echo "rc=$?" > $D/rc
