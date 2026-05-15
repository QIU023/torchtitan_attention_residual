#!/usr/bin/env bash
# Snapshot of all 8-GPU run states for quick inspection.
# Run with no args.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== running processes ==="
ps -ef | grep -E "phase5_vlm_multimodal_sft.train_mm|run_remaining|run_v10|run_tier_b_a" | grep -v grep | awk '{print $2, $11, $12}' | head -20

echo ""
echo "=== GPU state ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo ""
echo "=== alignment runs progress ==="
for d in phase5_vlm_multimodal_sft/runs/8gpu_*; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    log="$d/train.log"
    if [[ ! -f "$log" ]]; then
        echo "  $name: no train.log yet"
        continue
    fi
    last_step=$(sed 's/\x1b\[[0-9;]*m//g; s/\x1b\[[0-9;]*[ -\/]*[@-~]//g' "$log" 2>/dev/null \
        | grep -oE "step:[ ]+[0-9]+[ ]+loss:[ ]+[0-9.]+" | tail -1)
    finished=""
    if grep -q "Training completed" "$log" 2>/dev/null; then
        finished=" [DONE]"
    fi
    err=""
    if grep -qE "Traceback|FAILED|RuntimeError|ImportError" "$log" 2>/dev/null; then
        err=" [ERROR]"
    fi
    echo "  $name: $last_step$finished$err"
done

echo ""
echo "=== orchestrator log tail (last 5 lines) ==="
tail -5 phase6_upstream_pr_prep/orchestrator_8gpu.log 2>/dev/null
