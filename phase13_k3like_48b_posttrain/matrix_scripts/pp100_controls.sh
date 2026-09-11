#!/bin/bash
# PR 4312: the five 100-step controls that answer the numerics question, in PR
# 4500's protocol. Run on the REVIEW head (pp_review4, on upstream main
# d9ca9e55a), not on the PR head: the two are not bitwise with each other
# because of upstream #4347, so the evidence has to sit on the branch under
# review.
#
# Before running, apply pp100_probe_hacks.patch to the worktree (three
# uncommitted probe hacks: the KDA capability guard lifted for non-Blackwell
# cards, a kimi_k3_debugmodel_ppnaive alias that selects the whole-stack
# transport, and MB_REVERSE, which reverses the order the gradient
# accumulation groups are consumed in and changes nothing else).
#
# Usage: TITAN=<worktree> VENV=<venv> PYPRE=<attn-gym checkout> pp100_controls.sh
set -uo pipefail
MX=$(dirname "$0")/mx3.sh
export VENV=${VENV:-/workspace/venv_bfx9} PYPRE=${PYPRE:-/tmp/attn_gym_up}
export SEED_ROOT=${SEED_ROOT:-/workspace/.mx3_seeds_pp100} SEED_CFG=kimi_k3_debugmodel
export MEASURE_STEPS=100 WARM_STEPS=1
TITAN=${TITAN:?set TITAN to the worktree}
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
MB="--parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
# Four micro-batches per step: four accumulation groups for dp1, four pipeline
# micro-batches for the pipeline cells, so every cell sees the same tokens in
# the same number of pieces. TOKENS_PER_STEP=256 with TOKENS_PER_MB=64 matches
# PR 4500's token budget; the default 1024/256 is the larger-batch case. A
# micro-batch is not tied to the context length: trainer.py requires only that
# it divide by tp * (2 * cp), which is 1 for these cells.
TOKENS_PER_STEP=${TOKENS_PER_STEP:-1024}; TOKENS_PER_MB=${TOKENS_PER_MB:-256}
B="--training.num-tokens-per-train-step $TOKENS_PER_STEP --training.num-tokens-per-microbatch-per-dp-rank $TOKENS_PER_MB"

TRITON_CACHE_DIR=/tmp/triton_${TAG:-pp100}_a TITAN=$TITAN CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1
pp2|2|$D 1 $P 2 $MB
pp2vp2|2|$D 1 $P 2 $MB $IL" $MX ${TAG:-pp100}_a

TRITON_CACHE_DIR=/tmp/triton_${TAG:-pp100}_b TITAN=$TITAN CFG=kimi_k3_debugmodel_ppnaive BATCH="$B" \
CELLS="pp2vp2n|2|$D 1 $P 2 $MB $IL" $MX ${TAG:-pp100}_b

MB_REVERSE=1 TRITON_CACHE_DIR=/tmp/triton_${TAG:-pp100}_c TITAN=$TITAN CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1rev|1|$D 1" $MX ${TAG:-pp100}_c

echo PP100-DONE
