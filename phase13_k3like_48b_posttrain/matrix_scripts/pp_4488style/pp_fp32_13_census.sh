#!/bin/bash
# Step-1 gradient tensor dump (fp32, as trained) for the 13-layer fp32 pair, then the sign census.
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
MS=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
export VENV=/workspace/venv_bfx9 SEED_ROOT=/workspace/.mx3_seeds_pprt_fp32l13 SEED_CFG=kimi_k3_debugmodel13 PYPRE=/tmp/attn_gym_up MEASURE_STEPS=1 WARM_STEPS=1
export CUDA_VISIBLE_DEVICES=2,3 TRITON_CACHE_DIR=/workspace/.triton_pprt_fp32
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
G=/workspace/gt_fp32l13; rm -rf $G; mkdir -p $G
GRAD_TENSOR_DUMP=$G/dp1 TITAN=/tmp/wt_pprt CFG=kimi_k3_debugmodel13 BATCH="$B" CELLS="dp1|1|$D 1 $PD" $MX pprt_gt_dp1
GRAD_TENSOR_DUMP=$G/pp2 TITAN=/tmp/wt_pprt CFG=kimi_k3_debugmodel13 BATCH="$B" CELLS="pp2|2|$D 1 $P 2 --parallelism.num-pp-microbatches 2 $PD" $MX pprt_gt_pp2
ls -la $G
source $VENV/bin/activate; python $MS/pp_step10_census.py "dp1 vs pp2, fp32 masters, 13 layers, step 1" $G/dp1 $G/pp2
echo "PP-CENSUS-DONE"
