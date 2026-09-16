#!/bin/bash
# PP balance stack rerun on 63869c502 (import os restored), pp2 x vp2 off and on, seeded, one warm cache.
set -uo pipefail
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.num_pp_microbatches 16"
cd /workspace && TITAN=/tmp/wt_ppbal_4312 VENV=/workspace/venv_bfx9 PYPRE=$S:/workspace/pylib/attn_gym_main CFG=kimi_k3_debugmodel SEED_ROOT=/workspace/.mx4_seeds_4312 \
  BATCH="$B" CELLS="pp2_plain|2|ppx_probe|pp2_vp2_plain|
pp2_balance|2|ppx_probe|pp2_vp2_balance|" $M/mx4.sh ppbal2_4312
echo PPBAL2 DONE
