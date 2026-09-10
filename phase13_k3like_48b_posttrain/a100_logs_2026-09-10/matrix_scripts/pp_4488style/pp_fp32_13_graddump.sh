#!/bin/bash
# Step-1 per-parameter gradient dump (norm + sha1 before clip_grad_norm_) for the 13-layer fp32 pair; one mx3 invocation per cell so the dumps do not overwrite each other.
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 SEED_ROOT=/workspace/.mx3_seeds_pprt_fp32l13 SEED_CFG=kimi_k3_debugmodel13 PYPRE=/tmp/attn_gym_up MEASURE_STEPS=1 WARM_STEPS=1
export CUDA_VISIBLE_DEVICES=2,3 TRITON_CACHE_DIR=/workspace/.triton_pprt_fp32
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
G=/workspace/gd_fp32l13; rm -rf $G; mkdir -p $G
GRAD_DUMP=$G/dp1 TITAN=/tmp/wt_pprt CFG=kimi_k3_debugmodel13 BATCH="$B" CELLS="dp1|1|$D 1 $PD" $MX pprt_gd_dp1
GRAD_DUMP=$G/pp2 TITAN=/tmp/wt_pprt CFG=kimi_k3_debugmodel13 BATCH="$B" CELLS="pp2|2|$D 1 $P 2 --parallelism.num-pp-microbatches 2 $PD" $MX pprt_gd_pp2
ls -la $G
cat $G/pp2.rank0.step1.txt $G/pp2.rank1.step1.txt > $G/pp2.all.step1.txt
python3 /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/cmp_grad_dumps.py $G/dp1.rank0.step1.txt $G/pp2.all.step1.txt
echo "PP-GD-DONE"
