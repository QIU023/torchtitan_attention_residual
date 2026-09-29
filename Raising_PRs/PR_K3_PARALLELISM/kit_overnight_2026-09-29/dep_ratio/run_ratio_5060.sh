#!/bin/bash
# DEP ratio work on the 8 x 5060 (main session queue; one GPU job at a time):
# 1. microbenchmark: encode vs one text stage, dims 1024 and 2048, seq 1024 and 2048, one GPU;
# 2. smoke: DEP off / K2.5 / bubble on the three data levels at dim 1024, pp2 x vpp4 x tp2 x ep2 (4 GPUs), 10 steps,
#    printing each run's vision_dep plan line and rc. 5060 is PCIe: step times here are only indicative.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=$(cd "$(dirname "$0")" && pwd); V=/workspace/venv_0928; W=${W:-$S/wt_dep_new}; R=${R:-$S/dep_ratio}; mkdir -p $R
note() { echo "$(date +%H:%M:%S) $*" | tee -a $R/progress.txt; }
note "tree $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"
for dim in 1024 2048; do for seq in 1024 2048; do
  ( cd $W && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$K:. timeout 1800 $V/bin/python $K/microbench_ratio.py --dim $dim --seq $seq ) \
    > $R/microbench_d${dim}_s${seq}.txt 2>&1
  note "microbench dim $dim seq $seq rc=$?"
done; done
declare -A RES=([L1]=224 [L2]=448 [L3]=1024)
for L in L1 L2 L3; do
  C=$R/cache_$L; rm -rf $C; mkdir -p $C
  for cfg in w_dep_off w_dep_k25 w_dep_bubble; do
    D=$R/smoke_${L}_$cfg; rm -rf $D; mkdir -p $D
    ( cd $W && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$K:. DEPW_DIM=1024 DEPR_RES=${RES[$L]} DEPR_SEQ=2048 \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc timeout 2400 $V/bin/torchrun --nproc_per_node=4 \
      --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_ratio_local \
      --config $cfg --training.steps 10 --metrics.log-freq 1 --dump-folder $D/out > $D/run.log 2>&1 )
    rc=$?; rm -rf $D/out
    note "smoke $L (${RES[$L]} px) $cfg rc=$rc; $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -o 'vision_dep: .*' | head -1)"
  done
  rm -rf $C
done
note "dep ratio 5060 done"
