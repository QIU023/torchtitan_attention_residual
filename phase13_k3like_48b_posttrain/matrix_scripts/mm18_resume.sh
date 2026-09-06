#!/bin/bash
# The seven 8-GPU cells the 2026-09-05 run of mm18_matrix.sh did not reach (paused at 11/18 for the
# review round). Same tree (mm18_int 40bb5bbb7 + the local registry alias), seed, batch and flags.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up
export TITAN=/tmp/wt_int18 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRITON_CACHE_DIR=/workspace/.triton_mm18_c
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
P="--parallelism.pipeline_parallel_degree"; C="--parallelism.context_parallel_degree"
E="--parallelism.expert_parallel_degree"; MB="--parallelism.num-pp-microbatches 8"
S="--parallelism.spmd_backend spmd_types"; NOSP="--parallelism.no-enable-sequence-parallel"
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
CFG=kimi_k3_debugmodel CELLS="fsdp2_tp2_pp2|8|$D 2 $T 2 $P 2 $MB $S
ep2_fsdp2_tp2_pp2|8|$D 2 $E 2 $T 2 $P 2 $MB $S" $MX mm18_c8b
CFG=kimi_k3_debugmodel_cp2 CELLS="fsdp2_pp2_cp2|8|$D 2 $P 2 $MB
ep2_fsdp2_pp2_cp2|8|$D 2 $E 2 $P 2 $MB
fsdp2_tp2_cp2|8|$D 2 $T 2 $NOSP
tp2_pp2_cp2|8|$D 1 $T 2 $P 2 $MB $NOSP
ep2_fsdp2_tp2_cp2|8|$D 2 $E 2 $T 2 $NOSP" $MX mm18_c8cpb
echo "MM18 RESUME DONE"
