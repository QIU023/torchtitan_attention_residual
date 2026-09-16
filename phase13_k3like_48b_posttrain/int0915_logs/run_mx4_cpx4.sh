#!/bin/bash
# cp2 x tp2 (tower cut at threshold 96), both flavours, with the partitioned encode's fresh tensors derived from typed inputs;
# mx4 with the matrix's seed checkpoint and a copy of its warm inductor cache.
set -uo pipefail
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
d=/workspace/mx4_int0916_0916_043002
cd /workspace && TITAN=/tmp/wt_int0916_cpx VENV=/workspace/venv_bfx9 PYPRE=$S:/workspace/pylib/attn_gym_main CFG=kimi_k3_debugmodel \
  INDUCTOR_SEED_CACHE=$d/inductor \
  BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256" \
  CELLS="cp2_tp2_ag|4|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_tp2_min96|
cp2_tp2_ul|4|cpmm_probe|kimi_k3_mm_ulysses_cp2_tp2_min96|" $M/mx4.sh int0916_cpx4
