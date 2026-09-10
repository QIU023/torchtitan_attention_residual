#!/bin/bash
# The main QB experiment: dp8 x ep8 on the 8-GPU box, 100 steps, 8192 tokens per rank per step (65536 global), sign-step vs quantile balancing, probe flavors for the loads.
MX=/workspace/matrix_scripts/mx3.sh
export VENV=/workspace/venv PYPRE=/workspace/attn_gym_up MEASURE_STEPS=100 WARM_STEPS=1 SEED_ROOT=/workspace/.mx3_seeds_qb65 SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 65536 --training.num-tokens-per-microbatch-per-dp-rank 256"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRITON_CACHE_DIR=/workspace/.triton_qb100_dp8 TITAN=/workspace/wt_qb CFG=kimi_k3_debugmodel_probe BATCH="$B" CELLS="dp8_ep8_ctrl|8|$D 8 $E 8 $PD" $MX qb100_dp8ep8_ctrl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRITON_CACHE_DIR=/workspace/.triton_qb100_dp8 TITAN=/workspace/wt_qb CFG=kimi_k3_debugmodel_qb_probe BATCH="$B" CELLS="dp8_ep8_qb|8|$D 8 $E 8 $PD" $MX qb100_dp8ep8_qb
for d in /workspace/mx3_qb100_dp8ep8_*; do grep -a "seed-ok\|ABORT" $d/results.txt; done
echo "QB-DP8-DONE"
