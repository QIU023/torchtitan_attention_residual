#!/bin/bash
# The same tables on packed C4 (allenai/c4 en, streamed), 100 measured steps; waits for the debug-set run.
source /workspace/kit/h200/env.sh; R=/workspace/results/summary.txt
until grep -q "ALL_DONE" $R; do sleep 30; done
run_family(){ echo "== $(date -u +%FT%TZ) matrix $1" | tee -a $R; bash /workspace/kit/h200/moonep_matrix_h200.sh $1 > /workspace/results/matrix_$1.out 2>&1; OUTD=$(ls -d /workspace/mx3_moonep_$1_* | tail -1); cat $OUTD/results.txt | tee -a $R; }
run_family c4dp2; run_family c4dp2floor; run_family c4dp4; run_family c4dp4floor
echo "== $(date -u +%FT%TZ) C4_DONE" | tee -a $R
