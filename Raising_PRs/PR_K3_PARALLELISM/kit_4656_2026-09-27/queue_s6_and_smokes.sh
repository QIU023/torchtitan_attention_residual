#!/bin/bash
# After the 4656 cells: PR A against its new base 4656 on the production layout, then both composition smokes.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_pp_lowerbound_2026-09-26
until [ -f $S/acr/SC_DONE ]; do sleep 20; done
export LB_ROOT=$S/lb6 PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_DIM=2048 PPMEM_SEQ=2048 PPMEM_LPS=3
bash $K/campaign2.sh s6 10 8 "b4656=$S/wt_attnres" "pra=$S/wt_ppopt"
R=$S/lb6/smoke; mkdir -p $R
smoke() {  # <name> <tree>
  local D=$R/$1; rm -rf $D; mkdir -p $D/cache; cd $2
  PYTHONPATH=/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
  /workspace/venv_bfx9/bin/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --role rank --tee 3 -m torchtitan.train --module torchtitan_recipes.tests.b200 \
    --config kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4 --training.steps 10 --dump-folder $D/out > $D/run.log 2>&1
  echo "rc=$?" > $D/rc
  rm -rf $D/cache
}
smoke compose_4656 $S/wt_attnres
smoke compose_pra $S/wt_ppopt
echo done > $S/lb6/ALL_DONE
