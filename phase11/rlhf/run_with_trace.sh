#!/usr/bin/env bash
# Phase 11 RLHF run with NCCL trace capture.
#
# Wraps run_grpo_llava_caption.py with the same NCCL_DEBUG envs we
# used for phase 7 catalog + phase 11 inference traces, so the
# resulting log can be fed straight into:
#   phase7/extract_collectives.py
#   phase7/expand_to_flows.py
#   phase7/flows_to_ixia.py
#
# Output structure mirrors phase11/trace_*/ inference dirs:
#   phase11/rlhf/trace_grpo_llava/
#     nccl-rank-<host>-<pid>.log    # per-rank NCCL collective log
#     run.log                       # stdout/stderr of the run
#     recipe.json                   # config snapshot for reproducibility
#     collective_summary.csv        # post-extract
#     flows.csv                     # post-expand
#     ixia_config.json              # final IXIA-consumable artifact
#
# Usage:
#   bash phase11/rlhf/run_with_trace.sh [--text-only]
set -uo pipefail

WS=/root/torchtitan_attention_residual
TRACE_DIR=${TRACE_DIR:-$WS/phase11/rlhf/trace_grpo_llava}
MODEL_PATH=${MODEL_PATH:-$WS/phase11/hf_aligned_447m_step12500}
NUM_STEPS=${NUM_STEPS:-20}

mkdir -p "$TRACE_DIR"
rm -f "$TRACE_DIR"/nccl-rank-*.log "$TRACE_DIR"/*.csv* "$TRACE_DIR"/*.json

cat > "$TRACE_DIR/recipe.json" <<EOF
{
  "phase": 11,
  "task": "rlhf-grpo-llava-caption",
  "framework": "torchtitan + monarch + sglang",
  "engine": "sglang",
  "method": "grpo",
  "trainer_mesh": "FSDP=4 ranks 0-3",
  "generator_mesh": "TP=4 ranks 4-7",
  "model_path": "$MODEL_PATH",
  "num_steps": $NUM_STEPS,
  "extra_args": "$*"
}
EOF

# NCCL trace envs — match phase 7 catalog convention.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="$TRACE_DIR/nccl-rank-%h-%p.log"
export NCCL_DEBUG_SUBSYS=COLL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Make the python tree importable. We MUST include
# /usr/local/lib/python3.12/dist-packages here because that's where
# torch lives on this box (the venv has only torchstore + monarch +
# torchdata; torch itself is system-wide). Without explicitly
# propagating this to PYTHONPATH, Monarch's spawned worker
# subprocesses fall back on a sys.path that's missing /usr/local
# and aborts on first ``import torch.distributed.rpc``.
TORCH_SITE=/usr/local/lib/python3.12/dist-packages
export PYTHONPATH="$WS:$WS/torchtitan:$TORCH_SITE${PYTHONPATH:+:${PYTHONPATH}}"

cd "$WS"

echo "==> phase 11/rlhf/run_grpo_llava_caption  trace_dir=$TRACE_DIR"
timeout 1800 python3 phase11/rlhf/run_grpo_llava_caption.py \
    --model-path "$MODEL_PATH" \
    --num-steps "$NUM_STEPS" \
    "$@" 2>&1 | tee "$TRACE_DIR/run.log"
rc=$?
echo "==> rc=$rc"

# Post-process the NCCL logs into the same artifacts phase 7 produces.
n_logs=$(ls "$TRACE_DIR"/nccl-rank-*.log 2>/dev/null | wc -l)
if (( n_logs > 0 )); then
    echo "==> post-process $n_logs NCCL logs"
    python3 "$WS/phase7/extract_collectives.py" "$TRACE_DIR/" >/dev/null \
        && echo "    extract OK ($(wc -l < $TRACE_DIR/collective_summary.csv) rows)"
    python3 "$WS/phase7/expand_to_flows.py" "$TRACE_DIR/" --world-size 8 >/dev/null \
        && echo "    flows OK ($(wc -l < $TRACE_DIR/flows.csv) rows)"
    python3 "$WS/phase7/flows_to_ixia.py" "$TRACE_DIR/" --world-size 8 >/dev/null \
        && echo "    ixia OK"
else
    echo "    !!! 0 nccl-rank logs captured !!!"
fi

# Gzip nccl + flows + collective_summary to keep the dir small.
gzip -9 "$TRACE_DIR"/nccl-rank-*.log "$TRACE_DIR"/flows.csv "$TRACE_DIR"/collective_summary.csv 2>/dev/null

echo "==> done"
ls -la "$TRACE_DIR"/
exit $rc
