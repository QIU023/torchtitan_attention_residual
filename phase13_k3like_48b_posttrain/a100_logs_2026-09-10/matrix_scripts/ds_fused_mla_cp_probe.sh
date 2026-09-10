#!/bin/bash
# Does the packed MLA Ulysses kernel compose with overrides/fused_mla.py? The override is specific to
# DeepSeek-V3's Attention (ComplexRoPE), so the test runs deepseek_v3_debugmodel on the CP tree
# (61a73ca6c, /tmp/wt_dsfused with three probe flavors, all with activation checkpointing off: fused_mla mutates tensors that selective AC caches): dp1 and cp2 x {generic, packed} x {stock, fused}.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_ds SEED_CFG=deepseek_v3_debugmodel MODULE=deepseek_v3 TITAN=/tmp/wt_dsfused PYPRE=/tmp/attn_gym_up
export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_dsfused MEASURE_STEPS=3
D="--parallelism.data_parallel_shard_degree"; S="--parallelism.spmd_backend spmd_types"
F="--override.imports torchtitan.overrides.fused_mla.fused_mla"
DC="--training.disable_cuda_graphs"  # deepseek_v3 under the token budget refuses CUDA-graph capture (the ctl matrices ran it off too)
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
CFG=deepseek_v3_debugmodel_noac BATCH="$B" CELLS="dp1|1|$D 1 $S $DC
dp1_fused|1|$D 1 $S $F $DC" $MX dsfused_dp1
CFG=deepseek_v3_debugmodel_cp2_generic BATCH="$B" CELLS="cp2_generic|2|$D 1 $DC
cp2_generic_fused|2|$D 1 $F $DC" $MX dsfused_generic
CFG=deepseek_v3_debugmodel_cp2_packed BATCH="$B" CELLS="cp2_packed|2|$D 1 $DC
cp2_packed_fused|2|$D 1 $F $DC" $MX dsfused_packed
echo "DS FUSED PROBE DONE"
