#!/bin/bash
# After gpu_queue5: the DEP ratio microbenchmarks again (their first run cast the rope buffers to bf16 and failed).
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
until grep -q "fake moonep load rerun finished" $S/gpu_queue.txt 2>/dev/null; do sleep 20; done
W=$S/wt_dep_new; R=$S/dep_ratio
for dim in 1024 2048; do for seq in 1024 2048; do
  ( cd $W && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$K/dep_ratio:. timeout 1800 /workspace/venv_0928/bin/python $K/dep_ratio/microbench_ratio.py \
    --dim $dim --seq $seq ) > $R/microbench_d${dim}_s${seq}.txt 2>&1
  echo "$(date +%H:%M:%S) microbench rerun dim $dim seq $seq rc=$?" >> $R/progress.txt
done; done
echo "$(date +%H:%M:%S) queue: dep microbench rerun finished" >> $S/gpu_queue.txt
