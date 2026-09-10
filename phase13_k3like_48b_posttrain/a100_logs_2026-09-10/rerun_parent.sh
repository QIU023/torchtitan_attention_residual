#!/bin/bash
# tp1_parent again on the idle GPU 7: the parent tree refused the A100 before its guard was relaxed.
source /workspace/venv/bin/activate
export PYTHONPATH=/workspace/attn_gym_up TT=/workspace/wt_tp TT_PARENT=/workspace/wt_parent OUT=/workspace/tp_a100_out
cd /workspace/matrix_scripts/tp_a100 && . ./common.sh
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
cell tp1_parent $TT_PARENT 7 1 bf16 100 $B $D 1 $PD
echo PARENT-RERUN-DONE
