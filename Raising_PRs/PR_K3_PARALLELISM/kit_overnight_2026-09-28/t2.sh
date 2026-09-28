#!/bin/bash
# T2: PR A (torch-runtime part only, d37fb90f1) against main f35966713 on 8 x 5060, the s6 layout
# (93 layers, block 12, 3 per stage, dim 2048, seq 2048, M16, full AC, pp8 x vp4): 10-step memory and
# identity, and a step-8 trace, every run on its own copy of one warm cache (campaign2.sh).
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
KL=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_pp_lowerbound_2026-09-26
SHIM=$S/lbplan/pr5_torch_compat_shim.patch
export LB_ROOT=$S/overnight/t2; mkdir -p $LB_ROOT
M=$S/wt_s_main; P=$S/wt_pra_cat3
for t in $M $P; do git -C $t apply $SHIM || { echo "shim failed on $t" > $LB_ROOT/FAILED; exit 1; }; done
export PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_LPS=3 PPMEM_DIM=2048 PPMEM_SEQ=2048
bash $KL/campaign2.sh t2 10 8 "main=$M" "pra=$P"
for t in $M $P; do git -C $t checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; done
for t in $M $P; do echo "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)"; done > $LB_ROOT/worktrees_after.txt
rm -rf $LB_ROOT/cache*
echo done > $LB_ROOT/DONE
