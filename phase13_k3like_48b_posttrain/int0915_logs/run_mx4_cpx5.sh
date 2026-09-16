#!/bin/bash
# The RFC's three-axis cells with CP (8 GPUs each, tower cut at threshold 96): fsdp2 x cp2 x tp2, fsdp2 x cp2 x pp2, tp2 x cp2 x pp2.
set -uo pipefail
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
d=/workspace/mx4_int0916_0916_043002
cd /workspace && TITAN=/tmp/wt_int0916_cpx VENV=/workspace/venv_bfx9 PYPRE=$S:/workspace/pylib/attn_gym_main CFG=kimi_k3_debugmodel \
  INDUCTOR_SEED_CACHE=$d/inductor \
  BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256" \
  CELLS="fsdp2_cp2_tp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_tp2_min96|--parallelism.data_parallel_shard_degree 2
fsdp2_cp2_pp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_pp2_min96|--parallelism.data_parallel_shard_degree 2 --parallelism.num_pp_microbatches 8
tp2_cp2_pp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_pp2_min96|--parallelism.tensor_parallel_degree 2 --parallelism.no-enable-sequence-parallel --parallelism.num_pp_microbatches 16" $M/mx4.sh int0916_cpx5
