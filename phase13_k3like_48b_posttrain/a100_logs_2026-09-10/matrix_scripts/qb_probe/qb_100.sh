#!/bin/bash
# QB on qb_review4 (743cefe6a, main 65ba8a697): 100 steps in 4500's format. Probe flavors (chained load probe) give loads and losses;
# plain flavors on this commit and on the parent give the same-commit control.
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/attn_gym_up MEASURE_STEPS=100 WARM_STEPS=1 SEED_ROOT=/workspace/.mx3_seeds_qb65 SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
run() { local tag=$1 gpus=$2 tree=$3 cfg=$4 cells=$5; CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=/workspace/.triton_qb100_$tag TITAN=$tree CFG=$cfg BATCH="$B" CELLS="$cells" $MX qb100_$tag; }
echo "=== wave 1"
run parent_dp1 6 /tmp/wt_main65 kimi_k3_debugmodel "parent_dp1|1|$D 1 $PD" &
sleep 150
run this_dp1 7 /tmp/wt_qbrun4 kimi_k3_debugmodel "this_dp1|1|$D 1 $PD" &
run dp1_ctrl 0 /tmp/wt_qbrun4 kimi_k3_debugmodel_probe "dp1_ctrl|1|$D 1 $PD" &
run dp1_qb 1 /tmp/wt_qbrun4 kimi_k3_debugmodel_qb_probe "dp1_qb|1|$D 1 $PD" &
run dp2_ctrl 2,3 /tmp/wt_qbrun4 kimi_k3_debugmodel_probe "dp2_ctrl|2|$D 2 $PD" &
run dp2_qb 4,5 /tmp/wt_qbrun4 kimi_k3_debugmodel_qb_probe "dp2_qb|2|$D 2 $PD" &
wait
echo "=== wave 2"
run dp2ep2_ctrl 0,1 /tmp/wt_qbrun4 kimi_k3_debugmodel_probe "dp2_ep2_ctrl|2|$D 2 $E 2 $PD" &
run dp2ep2_qb 2,3 /tmp/wt_qbrun4 kimi_k3_debugmodel_qb_probe "dp2_ep2_qb|2|$D 2 $E 2 $PD" &
wait
for d in /workspace/mx3_qb100_*; do grep -a "seed-ok\|ABORT" $d/results.txt; done
echo "QB-100-DONE"
