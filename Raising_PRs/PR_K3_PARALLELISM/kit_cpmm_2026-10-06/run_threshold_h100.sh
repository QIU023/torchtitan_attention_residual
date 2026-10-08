#!/bin/bash
# #4380 threshold question on the H100. Recipe path (typechecking off: SAC + local compile), the released K3 tower on the
# debug text model, CP all-gather; the PR's tree at the default threshold (256: split) against a threshold no image
# reaches (main's path), at typical sizes: cc12m-test at native size (841 to 2124 patches, several images per
# micro-batch), 1008 px and 1456 px; CP2 then CP4; 10 steps after a 5-step warm-up of each group's shared cache.
# Then one profiled tower step at 1008 px, CP2, SAC, main's tree and the PR's. Everything sequential (timing).
K=/workspace/kit_cpmm; C=/workspace/tt_cpmm; O=/workspace/h100_thr
export KIT_ENV=/workspace/kit_setup/env_cu126.sh PP_PRE=/workspace/kit_setup/shims WARM_STEPS=5 STEPS=10
mkdir -p $O
B="CP_MODE=allgather,CP_TOWER=k3,CP_TYPECHECK=0"
for size in native 1008 1456; do
  case $size in native) D="CP_MAXP=4096";; 1008) D="CP_IMG_PX=1008";; 1456) D="CP_IMG_PX=1456,CP_SEQ=4096";; esac
  for cp in 2 4; do
    g=$(seq -s, 0 $((cp - 1)))
    $K/run_matrix2.sh $O/${size}_cp$cp split:$C:$g:$B,$D,CP_DEGREE=$cp,CP_MINP=256,WARM=w whole:$C:$g:$B,$D,CP_DEGREE=$cp,CP_MINP=1000000,WARM=w
    echo "$(date +%H:%M:%S) ${size}_cp$cp $(tail -3 $O/${size}_cp$cp/progress.txt | head -2 | tr '\n' ' ')" >> $O/chain.txt
  done
done
. /workspace/kit_setup/env_cu126.sh
for t in main:/workspace/tt_main pr:$C; do
  n=${t%%:*}; tree=${t#*:}
  ( cd $tree && env CASE=1008px AC=sac TAG=prof_${n}_1008_cp2 OUT=$O/prof_results.jsonl PROFILE=$O/prof \
      PYTHONPATH=/workspace/kit_setup/shims_noat:$K:. TORCHINDUCTOR_CACHE_DIR=/workspace/h100_tower/cache/ic \
      TRITON_CACHE_DIR=/workspace/h100_tower/cache/tc CUDA_VISIBLE_DEVICES=0,1 \
      timeout 900 torchrun --nproc_per_node=2 --master_port=29714 $K/tower_bench.py > $O/prof_$n.log 2>&1 )
  echo "$(date +%H:%M:%S) prof $n rc=$?" >> $O/chain.txt
done
echo "$(date +%H:%M:%S) THR_DONE" >> $O/chain.txt
