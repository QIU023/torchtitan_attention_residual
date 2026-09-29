#!/bin/bash
# After gpu_queue3: the DEP ratio microbenchmarks and the three-level DEP smoke on 5060.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
until grep -q "fake moonep load finished" $S/gpu_queue.txt 2>/dev/null; do sleep 20; done
bash $K/dep_ratio/run_ratio_5060.sh > $S/dep_ratio.log 2>&1
echo "$(date +%H:%M:%S) queue: dep ratio finished" >> $S/gpu_queue.txt
