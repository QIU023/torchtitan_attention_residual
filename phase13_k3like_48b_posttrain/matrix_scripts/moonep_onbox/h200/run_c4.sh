#!/bin/bash
# The same tables on packed C4 (allenai/c4 en, streamed), 100 measured steps.
# Phase 1 (dp2 families) runs at once on GPUs 2,3 next to the debug-set dp2 run on GPUs 0,1;
# phase 2 (dp4 families) waits for the debug-set chain's ALL_DONE written after the last RESTART.
source /workspace/kit/h200/env.sh; R=/workspace/results/summary.txt
run_family(){ echo "== $(date -u +%FT%TZ) matrix $1 (GPUs $CUDA_VISIBLE_DEVICES)" | tee -a $R; bash /workspace/kit/h200/moonep_matrix_h200.sh $1 > /workspace/results/matrix_$1.out 2>&1; OUTD=$(ls -d /workspace/mx3_moonep_$1_* | tail -1); cat $OUTD/results.txt | tee -a $R; }
export CUDA_VISIBLE_DEVICES=2,3
run_family c4dp2; run_family c4dp2floor
until awk '/RESTART/{n=0} /ALL_DONE/{n=1} END{exit !n}' $R; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0,1,2,3
run_family c4dp4; run_family c4dp4floor
echo "== $(date -u +%FT%TZ) C4_DONE" | tee -a $R
