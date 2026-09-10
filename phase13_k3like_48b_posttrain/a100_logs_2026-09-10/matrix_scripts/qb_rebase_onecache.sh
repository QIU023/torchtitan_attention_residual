#!/bin/bash
# QB on qb_release rebased onto upstream/main d263ca0a1 (/tmp/wt_qbrebase), gym b19162e: the sign-step
# control (kimi_k3_debugmodel) and quantile balancing (kimi_k3_debugmodel_qb), dp1 / dp2 / dp2 x ep2, on
# ONE compile cache -- the QB invocation starts from the control's inductor cache and shares its triton cache.
# partial_dtensor per cell: without the declarations PR, spmd_types has no layout for the multimodal inputs on main.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up
export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_qb4
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
PD="--parallelism.spmd_backend partial_dtensor"
CELLS="dp1|1|$D 1 $PD
dp2|2|$D 2 $PD
dp2_ep2|2|$D 2 $E 2 $PD"
TITAN=/tmp/wt_qbrebase CFG=kimi_k3_debugmodel BATCH="$B" CELLS="$CELLS" $MX qb4_control
CTRL=$(ls -td /workspace/mx3_qb4_control_* | head -1)
INDUCTOR_SEED_CACHE=$CTRL/inductor TITAN=/tmp/wt_qbrebase CFG=kimi_k3_debugmodel_qb BATCH="$B" CELLS="$CELLS" $MX qb4_qb
for f in $CTRL/results.txt $(ls -td /workspace/mx3_qb4_qb_* | head -1)/results.txt; do echo "== $f"; grep -v "^seed\|^tree" $f; done
echo "QB4 ONECACHE DONE"
