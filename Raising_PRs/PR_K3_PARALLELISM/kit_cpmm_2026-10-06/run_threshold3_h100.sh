#!/bin/bash
# #4380 threshold, packed micro-batches (recipe path, K3 tower, CP all-gather, PR tree split at 256 vs never): cc12m-test at
# native size and at 1008 px, whole samples packed into seq 8192; CP2 then CP4; sequential.
K=/workspace/kit_cpmm; C=/workspace/tt_cpmm; O=/workspace/h100_thr3
export KIT_ENV=/workspace/kit_setup/env_cu126.sh PP_PRE=/workspace/kit_setup/shims WARM_STEPS=5 STEPS=10
mkdir -p $O
B="CP_MODE=allgather,CP_TOWER=k3,CP_TYPECHECK=0,CP_PACK=1,CP_SEQ=8192"
for size in native 1008; do
  case $size in native) D="CP_MAXP=4096";; 1008) D="CP_IMG_PX=1008";; esac
  for cp in 2 4; do
    g=$(seq -s, 0 $((cp - 1)))
    $K/run_matrix2.sh $O/${size}_cp$cp split:$C:$g:$B,$D,CP_DEGREE=$cp,CP_MINP=256,WARM=w whole:$C:$g:$B,$D,CP_DEGREE=$cp,CP_MINP=1000000,WARM=w
    echo "$(date +%H:%M:%S) ${size}_cp$cp done" >> $O/chain.txt
  done
done
echo "$(date +%H:%M:%S) THR3_DONE" >> $O/chain.txt
