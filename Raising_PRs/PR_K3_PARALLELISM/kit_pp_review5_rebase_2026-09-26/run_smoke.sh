#!/bin/bash
# Two 10-step smokes on pp_review5 (+ uncommitted torch-compat shim): the composition cell, then pp4 x vpp4.
set -uo pipefail
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/smoke_pr5/runs; mkdir -p $R
cd /tmp/wt_pp_review5
run() {  # <name> <ngpu> <module> <config>
  local name=$1 n=$2 mod=$3 cfg=$4 D=$R/$1
  rm -rf $D; mkdir -p $D/cache
  PYTHONPATH=$S/smoke_pr5:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
  /workspace/venv_bfx9/bin/torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --role rank --tee 3 \
    -m torchtitan.train --module $mod --config $cfg --training.steps 10 --dump-folder $D/out > $D/run.log 2>&1
  echo "rc=$?" > $D/rc
}
run compose 8 torchtitan_recipes.tests.b200 kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4
run pp4vpp4 4 smoke_pr5 kimi_k3_pp4_vpp4
echo done > $R/ALL_DONE
