#!/bin/bash
# Sizing for the H100 PP proposal: the 5060 campaign's structure (93 layers, block 12, 3 per stage,
# M16) at pp4 x vp8, two widths each, to fit peak = a + b * dim per rank and extrapolate.
#  F: full AC, seq 2048, main (the heaviest tree of the stack table), dim 1024 and 1536.
#  N: no AC, seq 512 (the b1 balance structure), #4764 with nothing on, dim 768 and 1024.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/sizing/res; mkdir -p $R; SHIM=$S/lbplan/pr5_torch_compat_shim.patch
M=$S/wt_s_main; B=/tmp/wt_ppbal
for t in $M $B; do git -C $t apply $SHIM || { echo "shim failed on $t" >> $R/progress.txt; exit 1; }; done
export LB_ROOT=$R PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_LPS=3
cell() {  # <name> <tree> <gpus> <steps> VAR=val...
  local name=$1 tree=$2 gpus=$3 n=$4; shift 4
  mkdir -p $R/cache_$name
  env "$@" bash $S/sizing/run_lb4_5060.sh $name $tree $R/cache_$name $n $gpus
  echo "$(date +%H:%M:%S) $name $(grep -o 'rc=[0-9]*' $R/$name/train.log | tail -1)" >> $R/progress.txt
}
cell f_d1024 $M 0,1,2,3 4 PPMEM_DIM=1024 PPMEM_SEQ=2048 &
cell f_d1536 $M 4,5,6,7 4 PPMEM_DIM=1536 PPMEM_SEQ=2048 &
wait
cell n_d768 $B 0,1,2,3 4 PPMEM_DIM=768 PPMEM_SEQ=512 PPMEM_AC=none &
cell n_d1024 $B 4,5,6,7 4 PPMEM_DIM=1024 PPMEM_SEQ=512 PPMEM_AC=none &
wait
for t in $M $B; do git -C $t checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; done
for t in $M $B; do echo "$t $(git -C $t status --short | wc -l)"; done > $R/worktrees_after.txt
rm -rf $R/cache_*
echo done > $R/SIZING_DONE
