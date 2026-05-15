#!/usr/bin/env bash
# Kimi-Linear AttnRes GRPO run with NCCL trace capture (Option C: full bf16
# workaround stack — no fp8, all known-working envs set). Post-processes the
# NCCL logs into the same artifact set as phase7's catalog (collective
# summary, expanded flows, Ixia traffic config) so the trace is directly
# comparable to prior PPO/GRPO sum-digits + qwen3 captures.
#
# Parallelism setup (matches recipe.json):
#   Trainer mesh:    PP=1  DP=4 (FSDP shard)  TP=1  EP=1  CP=1   (ranks 0-3, 1 GPU each)
#   Generator mesh:  PP=1  DP=1  TP=4         EP=1  CP=1   (ranks 4-7, sharing 4 GPUs)
#   Cross-mesh:      torchstore push (trainer 4→storage) + pull (storage→generator TP-shard)
#
# Scale-out-relevant traffic = FSDP DP all-gather/reduce-scatter +
# torchstore RPC weight broadcast. TP=4 on generator is intra-node NVLink only.

set -uo pipefail
cd /workspace/torchtitan_attention_residual

WS=$PWD
TS=$(date -u +%Y%m%dT%H%M%S)
TRACE_DIR=${TRACE_DIR:-$WS/phase11_rlhf_grpo_infra/rlhf/trace_grpo_kimi_attnres_${TS}}
mkdir -p "$TRACE_DIR"

# ---- Option C: full bf16 workaround stack ----
export PYTHONPATH="${WS}/torchtitan:${WS}"
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_DISABLE_SHM_MM=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
# Engine kwargs are set in-runner: decode_attention_backend=torch_native,
# disable_cuda_graph=True. quantization stays None (bf16).

# ---- NCCL trace capture (per-rank logs in TRACE_DIR) ----
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="$TRACE_DIR/nccl-rank-%h-%p.log"
export NCCL_DEBUG_SUBSYS=COLL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NUM_STEPS=${NUM_STEPS:-60}     # 60 steps ~= 20 min @ 20s/step — covers warmup +
                                # representative steady state for trace stats
NUM_EP=${NUM_EP:-4}             # per-step episodes for GRPO
CKPT_DCP=${CKPT_DCP:-$WS/phase5_vlm_multimodal_sft/runs/mm_sft_447m_full/checkpoint/step-3100}
CKPT_HF=${CKPT_HF:-$WS/phase5_vlm_multimodal_sft/runs/mm_sft_447m_full/hf_step3100}

cat > "$TRACE_DIR/recipe.json" <<EOF
{
  "phase": 11,
  "task": "rlhf-grpo-kimi-attnres-vlm",
  "framework": "torchtitan + monarch + sglang",
  "engine": "sglang",
  "method": "grpo",
  "task_dataset": "LLaVA-Pretrain captions, BLEU-1 reward vs gold",
  "model_backbone": "Kimi-Linear 447M Block AttnRes (KDA + MLA + MoE)",
  "model_multimodal": "SigLIP-Base + MLP projector",
  "model_dtype": "bfloat16",
  "quantization": "none",
  "ckpt_dcp": "$CKPT_DCP",
  "ckpt_hf":  "$CKPT_HF",
  "num_steps": $NUM_STEPS,
  "num_episodes_per_step": $NUM_EP,
  "parallelism": {
    "trainer_mesh":   {"pp": 1, "dp": 4, "tp": 1, "ep": 1, "cp": 1, "ranks": "0-3 on GPU 0-3 (1 GPU each, FSDP shard)"},
    "generator_mesh": {"pp": 1, "dp": 1, "tp": 4, "ep": 1, "cp": 1, "ranks": "4-7 on GPU 4-7 (shared, lead actor TP=4)"},
    "grader_mesh":    {"pp": 1, "dp": 1, "tp": 1, "ep": 1, "cp": 1, "ranks": "spawn_procs() default — CPU-only mesh"},
    "cross_mesh":     "torchstore push (trainer 4 → store) + pull (store → generator TP shard) every step",
    "scale_out_relevant": "FSDP DP all-gather/reduce-scatter (trainer mesh) + torchstore RPC weight broadcast every step",
    "scale_out_NOT_relevant": "TP=4 all-reduce inside generator mesh (intra-node NVLink only)"
  },
  "workarounds_active": [
    "ATTNRES_MLA_FP32_FALLBACK=1 — fp32 eager MLA on prefill (workaround for flashinfer_mla bf16 NaN on large AttnRes activations)",
    "decode_attention_backend=torch_native — eager SDPA for decode MLA (same root cause)",
    "disable_cuda_graph=True — torch_native has no cuda_graph support",
    "SGLANG_DISABLE_SHM_MM=1 — inline pickle for multimodal payloads (avoid monarch-lifecycle SHM race)",
    "SGLANG_FP8_IGNORED_LAYERS=attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts — fp8 unused here but env is set for safety",
    "torchstore Controller monkeypatch — promote 5 sync @endpoint to async (monarch sync/async mix rejection)",
    "grader_mesh sys.path bootstrap — propagate phase11_rlhf_grpo_infra/rlhf for LlavaCaptionTask pickle",
    "AttnRes block-aggregation einsums → manual broadcast+sum (cuBLAS bypass; bf16 unaffected)"
  ],
  "ts_utc": "$TS"
}
EOF

cat "$TRACE_DIR/recipe.json"
echo
echo "==> launching GRPO with trace capture: $TRACE_DIR"
echo "==> ${NUM_STEPS} steps × ${NUM_EP} episodes/step,model=Kimi-Linear-447M-AttnRes (bf16)"

timeout 2400 /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$CKPT_DCP" \
    --hf-model-path "$CKPT_HF" \
    --num-steps "$NUM_STEPS" \
    --num-episodes-per-step "$NUM_EP" 2>&1 | tee "$TRACE_DIR/run.log"
rc=$?
echo "==> rc=$rc"

# ---- Post-process ----
n_logs=$(ls "$TRACE_DIR"/nccl-rank-*.log 2>/dev/null | wc -l)
echo "==> NCCL log count: $n_logs"
if (( n_logs > 0 )); then
    echo "==> phase7_nccl_traffic_catalog/extract_collectives.py"
    /usr/bin/python3 "$WS/phase7_nccl_traffic_catalog/extract_collectives.py" "$TRACE_DIR/" \
        && echo "   collective_summary.csv: $(wc -l < $TRACE_DIR/collective_summary.csv) rows"

    echo "==> phase7_nccl_traffic_catalog/expand_to_flows.py"
    /usr/bin/python3 "$WS/phase7_nccl_traffic_catalog/expand_to_flows.py" "$TRACE_DIR/" --world-size 8 \
        && echo "   flows.csv: $(wc -l < $TRACE_DIR/flows.csv) rows"

    echo "==> phase7_nccl_traffic_catalog/flows_to_ixia.py"
    /usr/bin/python3 "$WS/phase7_nccl_traffic_catalog/flows_to_ixia.py" "$TRACE_DIR/" \
        && echo "   ixia_config.json: $(wc -c < $TRACE_DIR/ixia_config.json) bytes"

    # Gzip raw NCCL logs to match prior trace dir convention
    gzip -f "$TRACE_DIR"/nccl-rank-*.log 2>/dev/null
    gzip -f "$TRACE_DIR"/collective_summary.csv "$TRACE_DIR"/flows.csv 2>/dev/null
fi

echo "==> done: $TRACE_DIR"
ls -la "$TRACE_DIR"
