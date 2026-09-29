#!/bin/bash
# After gpu_queue2 (smoke, early-wait probe): pp2 x vpp2 again with expandable segments on both trees
# (#4656 ran out of memory there through fragmentation at dim 2048), then the fake-MoonEP load run.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
until grep -q "early-wait probe finished" $S/gpu_queue.txt 2>/dev/null; do sleep 20; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R=$S/bound_pp2vp2_es STEPS=20 bash $K/pra_bound/run_bound.sh pp2vp2 > $S/bound_pp2vp2_es.log 2>&1
echo "$(date +%H:%M:%S) queue: pp2vp2 with expandable segments finished" >> $S/gpu_queue.txt
MODE=fake TREE=$S/wt_moonep_rb VENV=/workspace/venv_0928 OUT=$S/moonep_load GPUS=0,1,2,3 \
  bash $K/moonep/run_load.sh > $S/moonep_load.log 2>&1
echo "$(date +%H:%M:%S) queue: fake moonep load finished" >> $S/gpu_queue.txt
