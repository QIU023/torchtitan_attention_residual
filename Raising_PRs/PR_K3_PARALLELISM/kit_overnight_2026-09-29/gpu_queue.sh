#!/bin/bash
# The main session's GPU queue on the 5060 box: one job at a time.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
until grep -q "bound campaign done" $S/bound/progress.txt 2>/dev/null; do sleep 20; done
bash $K/smoke_4765_4764/smoke.sh > $S/smoke_4765.log 2>&1
echo "$(date +%H:%M:%S) queue: smoke finished" >> $S/gpu_queue.txt
# Queued later: PR A against PR A with the forward sends waited at the rank's next virtual stage (probe patch).
until grep -q "smoke done" $S/smoke_4765/progress.txt 2>/dev/null; do sleep 20; done
A=$S/wt_pra_v3 B=$S/wt_pra_early NA=pra NB=early R=$S/bound_early STEPS=20 bash $K/pra_bound/run_bound2.sh pp4vp2 pp4vp4 > $S/bound_early.log 2>&1
echo "$(date +%H:%M:%S) queue: early-wait probe finished" >> $S/gpu_queue.txt
