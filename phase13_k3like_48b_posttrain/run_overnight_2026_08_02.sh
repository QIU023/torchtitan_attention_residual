#!/bin/bash
# Overnight: LoRA across the parallelism combinations, then a GRPO minimal loop.
#
# LoRA is measured as it stands even though the implementation is slated for
# replacement by upstream's LoRAConverter (blocked on a sharding_config refactor,
# see LORA_CONVERTER_BLOCKER). Knowing which combinations are affected is what
# sets the bar the replacement has to clear.
#
# Every LoRA leg loads a WARM checkpoint. From a cold seed LoRA's B is zero, so
# grad_A is exactly zero and the adapter contributes nothing -- a cold check
# measures an inert model and reports it as clean, which is how the o_proj defect
# survived one. See MEASUREMENT_REGIMES.
set -u
ROOT=/workspace/torchtitan_attention_residual
TITAN=$ROOT/torchtitan
OUT=${OUT:-/workspace/overnight_0802}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate
LOG="$OUT/log.txt"; : > "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

FLAVOR=kimi_k3_mini_qlora
BASE="--module kimi_k3 --config $FLAVOR --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1"

say "=== warm checkpoint (LoRA B must be nonzero or the check is inert) ==="
W=$OUT/warm; rm -rf $W
timeout 3600 torchrun --nproc_per_node=1 --master_port=57000 -m torchtitan.train \
  $BASE --training.steps 3 --training.global-batch-size 2 --training.local-batch-size 2 \
  --parallelism.data_parallel_shard_degree 1 --checkpoint.enable --checkpoint.interval 3 \
  --dump-folder $W >>"$LOG" 2>&1
SP=$(find $W -maxdepth 3 -type d -name "step-3" | head -1)
say "warm=${SP:-MISSING}"
[ -z "${SP:-}" ] && { say "ABORT: no warm checkpoint"; exit 1; }
LOAD="--checkpoint.enable --checkpoint.initial-load-path $SP \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

PORT=57100
leg() {
  local name="$1" ngpu="$2" lbs="$3"; shift 3
  PORT=$((PORT+1)); rm -rf "$OUT/r_$name" "$OUT/$name.json"*
  say "--- $name (${ngpu} GPU) ---"
  GRADCHK_DUMP="$OUT/$name.json" timeout 3600 torchrun --nproc_per_node=$ngpu \
    --master_port=$PORT ../phase13_k3like_48b_posttrain/tp_trainer_grad_probe.py \
    $BASE $LOAD --training.steps 6 --training.global-batch-size 8 \
    --training.local-batch-size $lbs "$@" --dump-folder "$OUT/r_$name" \
    >>"$LOG" 2>&1
  [ -f "$OUT/$name.json" ] || say "    $name: NO DUMP (see log)"
}

say "=== LoRA: reference and the choose-3 set, with and without EP ==="
leg ref        1 2 --parallelism.data_parallel_shard_degree 1
leg ref_dp2    2 2 --parallelism.data_parallel_shard_degree 2
leg fsdp2      2 2 --parallelism.data_parallel_shard_degree 2
leg pp2        2 4 --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 2
leg cp2        2 2 --parallelism.data_parallel_shard_degree 1 --parallelism.context_parallel_degree 2
leg tp2        2 2 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2
leg fsdp2_pp2  4 4 --parallelism.data_parallel_shard_degree 2 --parallelism.pipeline_parallel_degree 2
leg fsdp2_cp2  4 2 --parallelism.data_parallel_shard_degree 2 --parallelism.context_parallel_degree 2
leg pp2_cp2    4 4 --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 2 \
                   --parallelism.context_parallel_degree 2
leg fsdp2_pp2_cp2 8 4 --parallelism.data_parallel_shard_degree 2 \
                   --parallelism.pipeline_parallel_degree 2 --parallelism.context_parallel_degree 2
leg ep2_fsdp2     2 2 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2
leg ep2_fsdp2_pp2 4 4 --parallelism.data_parallel_shard_degree 2 \
                   --parallelism.expert_parallel_degree 2 --parallelism.pipeline_parallel_degree 2
leg ep2_fsdp2_cp2 4 2 --parallelism.data_parallel_shard_degree 2 \
                   --parallelism.expert_parallel_degree 2 --parallelism.context_parallel_degree 2

say "=== comparison ==="
# The comparison lives in lora_perparam_compare.py, not in a heredoc here.
# A heredoc after a PIPELINE attaches to the last command, so the previous
# form fed the script to tee and left python3 reading nothing -- the table
# silently never printed, which looks exactly like a completed run.
python3 ../phase13_k3like_48b_posttrain/lora_perparam_compare.py "$OUT" 2>&1 | tee -a "$LOG"
say "=== done ==="
