#!/bin/bash
# QB dp8 x ep8, 100 steps, after the dp2 rerun releases the GPUs.
until grep -q DP2-RERUN-DONE /workspace/rerun_dp2.log 2>/dev/null; do sleep 60; done
source /workspace/venv/bin/activate; export PYTHONPATH=/workspace/attn_gym_up
rm -rf /workspace/mx3_qb100_*
timeout 7200 bash /workspace/matrix_scripts/qb_dp8ep8_a100.sh 2>&1 | tail -6
echo QB-RERUN-DONE
