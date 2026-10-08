#!/bin/bash
# #4380 threshold, second set (recipe path, K3 tower, CP all-gather, PR tree split at 256 vs never): several 1008 px images
# per micro-batch (seq 8192 packs about six), and one 1260 px image (90 x 90 = 8100 patches) near the single-image
# crossover; CP2 then CP4; sequential.
K=/workspace/kit_cpmm; C=/workspace/tt_cpmm; O=/workspace/h100_thr2
export KIT_ENV=/workspace/kit_setup/env_cu126.sh PP_PRE=/workspace/kit_setup/shims WARM_STEPS=5 STEPS=10
mkdir -p $O
B="CP_MODE=allgather,CP_TOWER=k3,CP_TYPECHECK=0"
for size in multi1008 1260; do
  case $size in multi1008) D="CP_IMG_PX=1008,CP_SEQ=8192";; 1260) D="CP_IMG_PX=1260,CP_SEQ=4096";; esac
  for cp in 2 4; do
    g=$(seq -s, 0 $((cp - 1)))
    $K/run_matrix2.sh $O/${size}_cp$cp split:$C:$g:$B,$D,CP_DEGREE=$cp,CP_MINP=256,WARM=w whole:$C:$g:$B,$D,CP_DEGREE=$cp,CP_MINP=1000000,WARM=w
    echo "$(date +%H:%M:%S) ${size}_cp$cp done" >> $O/chain.txt
  done
done
echo "$(date +%H:%M:%S) THR2_DONE" >> $O/chain.txt
