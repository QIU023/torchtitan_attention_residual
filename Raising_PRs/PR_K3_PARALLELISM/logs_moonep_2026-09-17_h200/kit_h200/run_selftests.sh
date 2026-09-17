#!/bin/bash
# MoonEP's own tests on NP ranks (default 2). Usage: NP=2 run_selftests.sh
source /workspace/kit/h200/env.sh; NP=${NP:-2}; cd /workspace/moonep
echo "moonep $(git log --oneline -1 | cut -c1-60) NP=$NP handle=$MOONEP_MEM_HANDLE_TYPE"
for t in test_planning test_dispatch test_combine test_e2e test_grad_reduce test_prefetch; do
  echo "### $t"
  timeout 1200 torchrun --nproc_per_node=$NP --master_port=$((41400+NP)) -m pytest -x -q tests/$t.py > /workspace/results/selftest_np${NP}_$t.log 2>&1; rc=$?
  grep -E "passed|failed|error|skipped" /workspace/results/selftest_np${NP}_$t.log | grep -v "^\[" | tail -2; echo "rc=$rc"
done
echo "### SELFTESTS_DONE NP=$NP"
