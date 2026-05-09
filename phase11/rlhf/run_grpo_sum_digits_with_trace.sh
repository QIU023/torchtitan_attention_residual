#!/usr/bin/env bash
# Production GRPO sum-digits run via SGLang + NCCL trace capture.
set -uo pipefail

WS=/root/torchtitan_attention_residual
TRACE_DIR=${TRACE_DIR:-$WS/phase11/rlhf/trace_grpo_sum_digits}
MODEL_PATH=${MODEL_PATH:-$WS/phase11/rlhf/qwen3_0_6b}
NUM_STEPS=${NUM_STEPS:-50}

mkdir -p "$TRACE_DIR"
rm -f "$TRACE_DIR"/nccl-rank-*.log "$TRACE_DIR"/*.csv* "$TRACE_DIR"/*.json

cat > "$TRACE_DIR/recipe.json" <<EOF
{
  "phase": 11,
  "task": "rlhf-grpo-sum-digits",
  "framework": "torchtitan + monarch + sglang",
  "engine": "sglang",
  "method": "grpo",
  "trainer_mesh": "FSDP=4 ranks 0-3",
  "generator_mesh": "TP=4 ranks 4-7 (lead/follower; rank 0 lead)",
  "model_path": "$MODEL_PATH",
  "model": "Qwen3-0.6B (HF)",
  "num_steps": $NUM_STEPS
}
EOF

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="$TRACE_DIR/nccl-rank-%h-%p.log"
export NCCL_DEBUG_SUBSYS=COLL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$WS:$WS/torchtitan:/usr/local/lib/python3.12/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"

cd "$WS"

echo "==> phase 11/rlhf/run_grpo_sum_digits  trace_dir=$TRACE_DIR num_steps=$NUM_STEPS"
timeout 7200 python3 phase11/rlhf/run_grpo_sum_digits.py \
    --model-path "$MODEL_PATH" \
    --num-steps "$NUM_STEPS" 2>&1 | tee "$TRACE_DIR/run.log"
rc=$?
echo "==> rc=$rc"

n_logs=$(ls "$TRACE_DIR"/nccl-rank-*.log 2>/dev/null | wc -l)
if (( n_logs > 0 )); then
    echo "==> post-process $n_logs NCCL logs"
    python3 "$WS/phase7/extract_collectives.py" "$TRACE_DIR/" >/dev/null \
        && echo "    extract OK ($(wc -l < $TRACE_DIR/collective_summary.csv) rows)"
    python3 "$WS/phase7/expand_to_flows.py" "$TRACE_DIR/" --world-size 8 >/dev/null \
        && echo "    flows OK ($(wc -l < $TRACE_DIR/flows.csv) rows)"
    python3 "$WS/phase7/flows_to_ixia.py" "$TRACE_DIR/" --world-size 8 >/dev/null \
        && echo "    ixia OK"
fi

gzip -9 "$TRACE_DIR"/nccl-rank-*.log "$TRACE_DIR"/flows.csv "$TRACE_DIR"/collective_summary.csv 2>/dev/null

exit $rc
