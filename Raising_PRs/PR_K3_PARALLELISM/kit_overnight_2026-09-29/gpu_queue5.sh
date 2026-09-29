#!/bin/bash
# After the DEP ratio run: the fake-MoonEP load cells again with the probe's forward wrapper fixed.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
until grep -q "dep ratio finished" $S/gpu_queue.txt 2>/dev/null; do sleep 20; done
rm -rf $S/moonep_load
MODE=fake TREE=$S/wt_moonep_rb VENV=/workspace/venv_0928 OUT=$S/moonep_load GPUS=0,1,2,3 \
  bash $K/moonep/run_load.sh > $S/moonep_load.log 2>&1
echo "$(date +%H:%M:%S) queue: fake moonep load rerun finished" >> $S/gpu_queue.txt
