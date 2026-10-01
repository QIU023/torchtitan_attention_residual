#!/bin/bash
# 10-01 night, DEP cost ratio on 8 x 5060, phase 1: encode time against one text stage of pp4 x vpp4's 16 stage split,
# one GPU, the widened debug text at dim 1024 (the microbench runs all 17 layers on one card; dim 2048 does not fit in
# 16 GB) with the tower not widened (DEPV_TOWER=base: 256 wide, 2 layers) at seq 2048 and 4096, and with the released K3
# tower (27 layers, 1024 wide) at seq 2048 for reference.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_overnight_2026-09-29/dep_ratio
V=/workspace/venv_0928; W=${W:-$S/wt_dep_1001}; R=${R:-$S/dep_1001}; mkdir -p $R; P=$R/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "phase1 tree $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"
for x in base:2048 base:4096 k3:2048; do
  tower=${x%%:*}; seq=${x#*:}
  ( cd $W && DEPV_TOWER=$tower CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$K:. timeout 1800 $V/bin/python $K/microbench_ratio.py \
    --dim 1024 --seq $seq --stages 16 ) > $R/microbench_${tower}_d1024_s${seq}.txt 2>&1
  note "microbench $tower seq $seq rc=$?"
done
note "phase1 done"; echo done > $R/PHASE1_DONE
