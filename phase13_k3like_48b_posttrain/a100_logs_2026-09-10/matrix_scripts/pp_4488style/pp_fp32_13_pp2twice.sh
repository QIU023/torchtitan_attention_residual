#!/bin/bash
# Control: the pp2 cell (13-layer fp32) a second time on fresh triton + inductor caches, step-1 gradient dump, compared with the first pp2 dump and with dp1.
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 SEED_ROOT=/workspace/.mx3_seeds_pprt_fp32l13 SEED_CFG=kimi_k3_debugmodel13 PYPRE=/tmp/attn_gym_up MEASURE_STEPS=1 WARM_STEPS=1
export CUDA_VISIBLE_DEVICES=2,3 TRITON_CACHE_DIR=/workspace/.triton_pprt_fp32_fresh2
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
G=/workspace/gd_fp32l13
GRAD_DUMP=$G/pp2b TITAN=/tmp/wt_pprt CFG=kimi_k3_debugmodel13 BATCH="$B" CELLS="pp2|2|$D 1 $P 2 --parallelism.num-pp-microbatches 2 $PD" $MX pprt_gd_pp2b
cat $G/pp2b.rank0.step1.txt $G/pp2b.rank1.step1.txt > $G/pp2b.all.step1.txt
echo "--- pp2 run A vs pp2 run B"; python3 /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/cmp_grad_dumps.py $G/pp2.all.step1.txt $G/pp2b.all.step1.txt
echo "--- dp1 vs pp2 run B"; python3 /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/cmp_grad_dumps.py $G/dp1.rank0.step1.txt $G/pp2b.all.step1.txt
echo "PP-PP2B-DONE"
