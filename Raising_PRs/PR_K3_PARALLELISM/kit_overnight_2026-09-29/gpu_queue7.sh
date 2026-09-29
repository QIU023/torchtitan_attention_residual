#!/bin/bash
# After gpu_queue6: DEP bubble and K2.5 at M16 on L2 (448 px), 3 steps, to see whether a larger M gives the bubble placement room.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
until grep -q "dep microbench rerun finished" $S/gpu_queue.txt 2>/dev/null; do sleep 20; done
W=$S/wt_dep_new; R=$S/dep_ratio; C=$R/cache_m16; rm -rf $C; mkdir -p $C
for cfg in w_dep_k25 w_dep_bubble; do
  D=$R/m16_L2_$cfg; rm -rf $D; mkdir -p $D
  ( cd $W && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$K/dep_ratio:. DEPW_DIM=1024 DEPR_RES=448 DEPR_SEQ=2048 \
    TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc timeout 2400 /workspace/venv_0928/bin/torchrun --nproc_per_node=4 \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_ratio_local \
    --config $cfg --training.steps 3 --metrics.log-freq 1 --parallelism.num-pp-microbatches 16 \
    --training.num-tokens-per-train-step 32768 --training.num-tokens-per-microbatch-per-dp-rank 2048 --dump-folder $D/out > $D/run.log 2>&1 )
  rc=$?; rm -rf $D/out
  echo "$(date +%H:%M:%S) m16 L2 $cfg rc=$rc; $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -o 'vision_dep: .*' | head -1)" >> $R/progress.txt
done
rm -rf $C
echo "$(date +%H:%M:%S) queue: dep m16 check finished" >> $S/gpu_queue.txt
