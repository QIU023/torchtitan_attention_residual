#!/bin/bash
# The matrix and the probes after the np2 self-tests and hot probe (which passed on 2026-09-17 06:55Z).
source /workspace/kit/h200/env.sh; R=/workspace/results/summary.txt
run_family(){ echo "== $(date -u +%FT%TZ) matrix $1" | tee -a $R; bash /workspace/kit/h200/moonep_matrix_h200.sh $1 > /workspace/results/matrix_$1.out 2>&1; OUTD=$(ls -d /workspace/mx3_moonep_$1_* | tail -1); cat $OUTD/results.txt | tee -a $R; }
run_family dp2; OUT2=$(ls -d /workspace/mx3_moonep_dp2_* | tail -1)
run_family dp2floor
echo "== $(date -u +%FT%TZ) grad probe np2" | tee -a $R; SEED=$OUT2/seed NP=2 bash /workspace/kit/h200/run_grad_probe.sh 2>&1 | tee -a $R
run_family dp4; OUT4=$(ls -d /workspace/mx3_moonep_dp4_* | tail -1)
run_family dp4floor
echo "== $(date -u +%FT%TZ) selftests np4" | tee -a $R; NP=4 bash /workspace/kit/h200/run_selftests.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) hot probe np4" | tee -a $R; NP=4 bash /workspace/kit/h200/run_hot_probe.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) grad probe np4" | tee -a $R; SEED=$OUT4/seed NP=4 bash /workspace/kit/h200/run_grad_probe.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) ALL_DONE" | tee -a $R
