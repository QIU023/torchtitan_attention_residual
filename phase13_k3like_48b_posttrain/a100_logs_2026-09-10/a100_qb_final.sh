#!/bin/bash
cd /workspace
echo "=== qb ep 100 $(date +%T)"
rm -rf /workspace/mx3_qb100_*
timeout 7200 bash /workspace/matrix_scripts/qb_dp8ep8_a100.sh 2>&1 | tail -6
echo "=== done $(date +%T)"; echo QB-FINAL-DONE
