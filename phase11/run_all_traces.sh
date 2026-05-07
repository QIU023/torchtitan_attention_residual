#!/usr/bin/env bash
# Phase 11 — re-capture all inference + PPO traces post-optimization.
#
# Inference traces use the canonical aligned 447M ckpt (Kimi) and the
# Qwen3 dense 96.7M ckpt. Each is captured under TP=8 (pure TP fabric)
# and TP=2×PP=2×EP=2 3D (production target), with seq_shard={0,1} to
# show the fabric pattern delta from the SGLang-AttnRes RS+AG path.
# PPO trace re-uses phase 9-B's vLLM-free toy MLP (no ckpt dependency).
#
# Output:
#   phase11/trace_kimi_tp8_shard{0,1}/
#   phase11/trace_kimi_3d_shard{0,1}/
#   phase11/trace_qwen3_3d_shard1/
#   phase5/runs/ppo_smoke_no_vllm/tier_b_trace/    (re-run, in-place)
set -uo pipefail

WS=/root/torchtitan_attention_residual
KIMI_MODEL=$WS/phase11/hf_aligned_447m
QWEN3_MODEL=$WS/phase11/hf_qwen3_attn_res

if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    source /venv/main/bin/activate
fi

postprocess() {
    local trace_dir="$1"
    local n_logs=$(ls "$trace_dir"/nccl-rank-*.log 2>/dev/null | wc -l)
    if (( n_logs > 0 )); then
        python3 "$WS/phase7/extract_collectives.py" "$trace_dir/" >/dev/null 2>&1 \
            && echo "    extract OK ($(wc -l < $trace_dir/collective_summary.csv) rows)"
        python3 "$WS/phase7/expand_to_flows.py" "$trace_dir/" --world-size 8 >/dev/null 2>&1 \
            && echo "    flows OK ($(wc -l < $trace_dir/flows.csv) rows)"
        python3 "$WS/phase7/flows_to_ixia.py" "$trace_dir/" --world-size 8 >/dev/null 2>&1 \
            && echo "    ixia OK"
    else
        echo "    !!! 0 nccl-rank logs captured !!!"
    fi
}

run_inference_trace() {
    local trace_dir="$1"
    local model="$2"
    local tp="$3"
    local pp="$4"
    local ep="$5"
    local shard="$6"
    local backend_extra="$7"

    echo "==> $(basename "$trace_dir") tp=$tp pp=$pp ep=$ep shard=$shard"
    mkdir -p "$trace_dir"
    rm -f "$trace_dir"/nccl-rank-*.log "$trace_dir"/*.csv* "$trace_dir"/*.json

    cat > "$trace_dir/recipe.json" <<EOF
{"phase":11,"mesh":{"TP":$tp,"PP":$pp,"EP":$ep},"world_size":$((tp*pp)),
"seq_shard":$shard,"model":"$(basename $model)","framework":"sglang",
"workload":"engine_init + 5 generations × 8 tokens"}
EOF

    cd /sgl-workspace/sglang
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_FILE="$trace_dir/nccl-rank-%h-%p.log" \
    NCCL_DEBUG_SUBSYS=COLL \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    SGLANG_ATTN_RES_SEQ_SHARD=$shard \
    timeout 600 python3 -c "
import sglang as sgl
e = sgl.Engine(
    model_path='$model',
    skip_tokenizer_init=True, tp_size=$tp, pp_size=$pp, ep_size=$ep,
    dtype='bfloat16', mem_fraction_static=0.5,
    disable_cuda_graph=True, disable_piecewise_cuda_graph=True, log_level='error',
    $backend_extra
)
print('BOOT_OK', flush=True)
for i in range(5):
    out = e.generate(input_ids=[[1,2,3,4,5,6,7,8]], sampling_params={'max_new_tokens':8,'temperature':0})
    print(f'gen {i}: {out[0][\"output_ids\"]}', flush=True)
e.shutdown()
" >"$trace_dir/run.log" 2>&1
    rc=$?
    cd "$WS"
    echo "    boot+gen rc=$rc"
    postprocess "$trace_dir"
}

# ---------- Kimi AttnRes inference traces ----------
KIMI_BACKEND="attention_backend='flashinfer', linear_attn_backend='triton',"

run_inference_trace "$WS/phase11/trace_kimi_tp8_shard0" "$KIMI_MODEL" 8 1 1 0 "$KIMI_BACKEND"
run_inference_trace "$WS/phase11/trace_kimi_tp8_shard1" "$KIMI_MODEL" 8 1 1 1 "$KIMI_BACKEND"
run_inference_trace "$WS/phase11/trace_kimi_3d_shard0" "$KIMI_MODEL"  2 2 2 0 "$KIMI_BACKEND"
run_inference_trace "$WS/phase11/trace_kimi_3d_shard1" "$KIMI_MODEL"  2 2 2 1 "$KIMI_BACKEND"

# ---------- Qwen3 cross-carrier verification ----------
QWEN3_BACKEND="attention_backend='flashinfer',"
run_inference_trace "$WS/phase11/trace_qwen3_3d_shard1" "$QWEN3_MODEL" 2 2 2 1 "$QWEN3_BACKEND"

# ---------- Phase 9-B PPO toy (re-run, random-init MLP, no ckpt) ----------
echo ""
echo "==> phase 9-B PPO smoke (re-run for fresh trace)"
PPO_OUT=$WS/phase5/runs/ppo_smoke_no_vllm
PPO_TRACE=$PPO_OUT/tier_b_trace
mkdir -p "$PPO_TRACE"
rm -f "$PPO_TRACE"/nccl-rank-*.log "$PPO_TRACE"/*.csv* "$PPO_TRACE"/*.json

# Decompress old gz logs/scripts can ignore — wipe.
rm -f "$PPO_TRACE"/nccl-rank-*.log.gz "$PPO_TRACE"/*.csv.gz "$PPO_TRACE"/run.log

NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE="$PPO_TRACE/nccl-rank-%h-%p.log" \
NCCL_DEBUG_SUBSYS=COLL \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
PYTHONPATH="$WS:${PYTHONPATH:-}" \
torchrun --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    -m phase9.ppo_smoke_no_vllm \
    >"$PPO_OUT/run.log" 2>&1
rc=$?
echo "    boot+gen rc=$rc"
postprocess "$PPO_TRACE"

echo ""
echo "=== ALL DONE ==="
ls -la "$WS"/phase11/trace_*/ixia_config.json 2>/dev/null | awk '{print "  "$NF}'
ls -la "$PPO_TRACE"/ixia_config.json 2>/dev/null | awk '{print "  "$NF}'
