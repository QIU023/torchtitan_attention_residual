#!/bin/bash
# The whole H200 MoonEP smoke, in order. Waits for the environment setup to finish.
mkdir -p /workspace/results; R=/workspace/results/summary.txt
until grep -q "SETUP_UV_DONE" /workspace/setup_uv.log; do sleep 20; done
grep -q "MOONEP_OK" /workspace/setup_uv.log || { echo "MOONEP build/import failed; stop" | tee -a $R; exit 1; }
source /workspace/kit/h200/env.sh
echo "== $(date -u +%FT%TZ) cpu tests" | tee -a $R
( cd $TITAN && timeout 900 python -m pytest -q -p no:cacheprovider tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py > /workspace/results/cpu_tests.log 2>&1; echo "cpu tests rc=$? $(tail -1 /workspace/results/cpu_tests.log)" ) | tee -a $R
echo "== $(date -u +%FT%TZ) selftests np2" | tee -a $R; NP=2 bash /workspace/kit/h200/run_selftests.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) hot probe np2" | tee -a $R; NP=2 bash /workspace/kit/h200/run_hot_probe.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) matrix" | tee -a $R; bash /workspace/kit/h200/moonep_matrix_h200.sh moonep 2>&1 | tail -12 | tee -a $R
OUT=$(ls -d /workspace/mx3_moonep_* | tail -1); cat $OUT/results.txt | tee -a $R
echo "== $(date -u +%FT%TZ) grad probe np2" | tee -a $R; SEED=$OUT/seed NP=2 bash /workspace/kit/h200/run_grad_probe.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) selftests np4" | tee -a $R; NP=4 bash /workspace/kit/h200/run_selftests.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) hot probe np4" | tee -a $R; NP=4 bash /workspace/kit/h200/run_hot_probe.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) grad probe np4" | tee -a $R; SEED=$OUT/seed NP=4 bash /workspace/kit/h200/run_grad_probe.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) ALL_DONE" | tee -a $R
