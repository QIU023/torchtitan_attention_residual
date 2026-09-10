#!/bin/bash
# QB on qb_review4 (743cefe6a, main 65ba8a697): 100 steps in 4500's format. Probe flavors (chained load probe) give loads and losses;
# plain flavors on this commit and on the parent give the same-commit control.
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/attn_gym_up MEASURE_STEPS=10 WARM_STEPS=1 SEED_ROOT=/workspace/.mx3_seeds_qb65 SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
run() { local tag=$1 gpus=$2 tree=$3 cfg=$4 cells=$5; CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=/workspace/.triton_qb100_$tag TITAN=$tree CFG=$cfg BATCH="$B" CELLS="$cells" $MX qb10_$tag; }
echo "=== wave 1"
run parent_dp1 6 /tmp/wt_main65 kimi_k3_debugmodel "parent_dp1|1|$D 1 $PD" &
sleep 150
run this_dp1 7 /tmp/wt_qbrun4 kimi_k3_debugmodel "this_dp1|1|$D 1 $PD" &
run dp1_ctrl 0 /tmp/wt_qbrun4 kimi_k3_debugmodel_probe "dp1_ctrl|1|$D 1 $PD" &
run dp1_qb 1 /tmp/wt_qbrun4 kimi_k3_debugmodel_qb_probe "dp1_qb|1|$D 1 $PD" &
run dp2_ctrl 2,3 /tmp/wt_qbrun4 kimi_k3_debugmodel_probe "dp2_ctrl|2|$D 2 $PD" &
run dp2_qb 4,5 /tmp/wt_qbrun4 kimi_k3_debugmodel_qb_probe "dp2_qb|2|$D 2 $PD" &
wait
echo "=== wave 2 (+ frozen pair)"
run dp2ep2_ctrl 0,1 /tmp/wt_qbrun4 kimi_k3_debugmodel_probe "dp2_ep2_ctrl|2|$D 2 $E 2 $PD" &
run dp2ep2_qb 2,3 /tmp/wt_qbrun4 kimi_k3_debugmodel_qb_probe "dp2_ep2_qb|2|$D 2 $E 2 $PD" &
run frozen_ctrl 4 /tmp/wt_qbrun4 kimi_k3_debugmodel_probe_frozen "dp1_frozen_ctrl|1|$D 1 $PD" &
run frozen_qb 5 /tmp/wt_qbrun4 kimi_k3_debugmodel_qb_probe_frozen "dp1_frozen_qb|1|$D 1 $PD" &
wait
echo "=== wave 3: dp8 x ep8"
B8="--training.num-tokens-per-train-step 65536 --training.num-tokens-per-microbatch-per-dp-rank 256"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRITON_CACHE_DIR=/workspace/.triton_qb100_dp8 TITAN=/tmp/wt_qbrun4 CFG=kimi_k3_debugmodel_probe BATCH="$B8" CELLS="dp8_ep8_ctrl|8|$D 8 $E 8 $PD" $MX qb10_dp8ep8_ctrl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRITON_CACHE_DIR=/workspace/.triton_qb100_dp8 TITAN=/tmp/wt_qbrun4 CFG=kimi_k3_debugmodel_qb_probe BATCH="$B8" CELLS="dp8_ep8_qb|8|$D 8 $E 8 $PD" $MX qb10_dp8ep8_qb
for d in /workspace/mx3_qb10_*; do grep -a "seed-ok\|ABORT" $d/results.txt; done
source /workspace/venv_bfx9/bin/activate; python3 /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/qb_summarize.py --short
echo "QB-10-DONE"
