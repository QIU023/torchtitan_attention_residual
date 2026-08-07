#!/bin/bash
# Does the LoRA regime itself introduce the PP/CP/TP gradient deviation?
#
# The per-parameter LoRA matrix shows 0.14-0.23 on PP, CP and TP, concentrated in
# the AttnRes pseudo-query projections -- and that is NOT a near-zero-gradient
# metric artifact: restricted to parameters holding more than 1% of the gradient
# norm it is still 0.14-0.23, and the worst offender (mlp_res_proj) holds 60% of
# the norm under LoRA because the frozen base contributes none.
#
# The full-parameter model was recorded as 0.00000 on PP, but on a different
# flavor, so "LoRA is what breaks it" was an inference. This runs BOTH arms on
# flavors that differ in exactly one field, lora_rank:
#   kimi_k3_mini_qlora             = kimi_k3_mini_block_attn_res + lora_rank 8
#
# Both arms at bfloat16, because at fp32 without FSDP's mixed-precision cast the
# full-parameter arm dies in fla's KDA kernel (asks 108160 bytes of dynamic
# shared memory against this card's 101376) while the LoRA arm does not. Forcing
# both to bf16 keeps lora_rank the only difference; leaving the default would
# make dtype a second one.
set -u
ROOT=/workspace/torchtitan_attention_residual
TITAN=$ROOT/torchtitan
OUT=${OUT:-/workspace/lora_vs_full}
STEPS=${STEPS:-6}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate
PORT=58100

arm() {
  local tag="$1" flavor="$2"
  local base="--module kimi_k3 --config $flavor --training.seq_len 512 \
   --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
   --training.dtype bfloat16"
  echo "########## arm $tag ($flavor) ##########"

  # Warm start: from a cold seed LoRA's B is zero, so grad_A is exactly zero and
  # the adapter contributes nothing -- a cold check measures an inert model and
  # reports it clean. Same protocol on both arms so the comparison holds.
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
  leg tp2 2 2 --parallelism.data_parallel_shard_degree 1 \
              --parallelism.tensor_parallel_degree 2
}

arm lora kimi_k3_mini_qlora
arm full kimi_k3_mini_block_attn_res
echo "########## DONE ##########"
