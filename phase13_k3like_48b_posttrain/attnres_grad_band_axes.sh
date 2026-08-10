#!/usr/bin/env bash
# Is the per-parameter gradient band a precision artifact of the AttnRes graft?
#
# Standing (LORA_PARALLELISM_MATRIX_2026-08-02, after the axes rerun): the band is NOT
# LoRA's -- the full-parameter arm is in the same norm-weighted band and an order worse
# unrestricted -- and the worst-weighted parameter is an AttnRes pseudo-query in both
# arms. It was left as "still unexplained, and it is now an AttnRes question".
#
# The hypothesis this tests. The pseudo-query gradient is a difference of nearly equal
# terms: attn_res.py records 6x to 15x cancellation, measured, which is why alpha is
# computed in fp32 there. Cancellation amplifies the RELATIVE effect of any change in
# summation order, and a different parallelism structure is exactly a different
# summation order. If that is the mechanism, the band is precision and not a defect,
# and it must collapse when the inputs to the cancellation are fp32.
#
# Two variables, one at a time, so neither answer needs a second document:
#   dtype     bfloat16 vs float32
#   AttnRes   present vs absent (the _noattnres flavor is the same model otherwise)
#
# fp32 is reachable here only because these flavors are MLA-only: with KDA, fla's kernel
# asks 108160 bytes of dynamic shared memory against this card's 101376 and fp32 dies.
# That is why this uses diag_4l_mla rather than the mini flavor the earlier runs used.
#
# Predictions, written before running:
#   attnres  x bf16   band present
#   attnres  x fp32   band collapses          <- the load-bearing cell
#   noattnres x bf16  no band
#   noattnres x fp32  no band
# A band that survives fp32 WITH AttnRes, or appears WITHOUT it, refutes the hypothesis
# rather than needing a patch to the story.
set -u
ROOT=/workspace/torchtitan_attention_residual
TITAN=$ROOT/torchtitan
OUT=${OUT:-/workspace/attnres_band}
STEPS=${STEPS:-6}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate
PORT=58600

arm() {
  local tag="$1" flavor="$2" dtype="$3"
  local base="--module kimi_k3 --config $flavor --training.seq_len 512 \
   --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
   --training.dtype $dtype"
  echo "########## arm $tag ($flavor, $dtype) ##########"

  # Warm start, same protocol as lora_vs_fullparam_axes: a cold seed leaves the
  # pseudo-queries at exactly zero, and their gradient structure at that point is not
  # the one the band was measured on.
  local w="$OUT/${tag}_warm"; rm -rf "$w"
  PORT=$((PORT+1))
  timeout 3600 torchrun --nproc_per_node=1 --master_port=$PORT -m torchtitan.train \
    $base --training.steps 3 --training.global-batch-size 2 \
    --training.local-batch-size 2 --parallelism.data_parallel_shard_degree 1 \
    --checkpoint.enable --checkpoint.interval 3 --dump-folder "$w" \
    > "$OUT/${tag}_warm.log" 2>&1
  local sp; sp=$(find "$w" -maxdepth 3 -type d -name "step-3" | head -1)
  if [ -z "${sp:-}" ]; then
    echo "  ABORT $tag: no warm checkpoint"
    grep -oiE "(RuntimeError|InternalError|ValueError|AssertionError): .{0,90}" \
      "$OUT/${tag}_warm.log" | head -2
    return 1
  fi
  echo "  warm=$sp"
  local load="--checkpoint.enable --checkpoint.initial-load-path $sp \
   --checkpoint.initial-load-model-only --checkpoint.interval 100000"

  leg() {
    local name="$1" ngpu="$2" lbs="$3"; shift 3
    PORT=$((PORT+1)); rm -rf "$OUT/r_${tag}_$name" "$OUT/${tag}_$name.json"*
    GRADCHK_DUMP="$OUT/${tag}_$name.json" timeout 3600 torchrun \
      --nproc_per_node=$ngpu --master_port=$PORT \
      "$ROOT/phase13_k3like_48b_posttrain/tp_trainer_grad_probe.py" \
      $base $load --training.steps $STEPS --training.global-batch-size 8 \
      --training.local-batch-size $lbs "$@" --dump-folder "$OUT/r_${tag}_$name" \
      > "$OUT/${tag}_$name.log" 2>&1
    if [ -f "$OUT/${tag}_$name.json.r0" ]; then echo "  $name ok"; else
      echo "  $name NO DUMP"
      grep -oiE "(RuntimeError|InternalError|ValueError|AssertionError): .{0,90}" \
        "$OUT/${tag}_$name.log" | head -2
    fi
  }

  leg ref 1 2 --parallelism.data_parallel_shard_degree 1
  leg pp2 2 4 --parallelism.data_parallel_shard_degree 1 \
              --parallelism.pipeline_parallel_degree 2
  leg cp2 2 2 --parallelism.data_parallel_shard_degree 1 \
              --parallelism.context_parallel_degree 2
}

arm nokda_bf16 kimi_k3_mini_diag_no_kda          bfloat16
arm mini_bf16  kimi_k3_mini_block_attn_res       bfloat16
echo "########## DONE ##########"
