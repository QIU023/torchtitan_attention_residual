#!/usr/bin/env bash
# Post-SFT VLM pipeline validation. Run after the LLaVA-Instruct-150K
# SFT completes. Steps:
#
#   1. Convert the SFT DCP ckpt → HF safetensors via the VLM converter.
#   2. Boot SGLang Engine on TP=1 with KimiAttnResVLForConditionalGeneration.
#   3. Decode one image+text prompt to verify end-to-end correctness.
#
# Single-GPU smoke (CUDA_VISIBLE_DEVICES=7) so it does not collide with
# any concurrent multi-GPU work. A successful run proves the entire
# VLM serving stack: DCP → HF safetensors → SGLang VLM model class →
# SigLIP → projector → AttnRes LM → tokenizer roundtrip.
set -uo pipefail

WS=/root/torchtitan_attention_residual
SFT_DIR="$WS/phase5/runs/sft_v_fsdp8_447m_aligned_llava_instruct_150k/checkpoint"

# Auto-detect the latest step-N ckpt. Watchdog keeps only the latest
# while training, so this is usually a single dir.
if [[ -z "${SFT_CKPT:-}" ]]; then
    latest=$(ls -1 "$SFT_DIR" 2>/dev/null | grep -E '^step-[0-9]+$' \
        | sort -t- -k2 -n | tail -1)
    if [[ -z "$latest" ]]; then
        echo "ERROR: no step-N ckpt found under $SFT_DIR"
        exit 1
    fi
    SFT_CKPT="$SFT_DIR/$latest"
fi
LATEST_STEP=$(basename "$SFT_CKPT" | grep -oE '[0-9]+$')
HF_OUT="${HF_OUT:-$WS/phase11/hf_aligned_447m_vlm_sft${LATEST_STEP}}"

echo "==> using SFT ckpt: $SFT_CKPT (step $LATEST_STEP)"
echo "==> HF output dir : $HF_OUT"
echo

if [[ ! -d "$SFT_CKPT" ]]; then
    echo "ERROR: SFT ckpt not found at $SFT_CKPT"
    exit 1
fi

echo "==> 1. converting DCP -> HF VLM safetensors"
echo "    in:  $SFT_CKPT"
echo "    out: $HF_OUT"
mkdir -p "$HF_OUT"
torchrun --nproc_per_node=1 --master-port=29521 \
    "$WS/phase11/dcp_to_hf_kimi_attn_res_vl.py" \
    --in "$SFT_CKPT" \
    --out "$HF_OUT" \
    --config kimi_linear_447m_aligned_block_attn_res_n4 \
    --vision-tower google/siglip-base-patch16-224 \
    --vision-hidden-size 768

echo
echo "==> 2. SGLang Engine smoke (TP=1 on GPU 7)"
CUDA_VISIBLE_DEVICES=7 timeout 360 python3 "$WS/phase11/smoke_vlm_engine.py" \
    --model-path "$HF_OUT" \
    --tp-size 1

echo
echo "==> 3. Qualitative eval on 10 LLaVA-Pretrain images (TP=1 on GPU 7)"
CUDA_VISIBLE_DEVICES=7 timeout 600 python3 "$WS/phase11/eval_vlm_qualitative.py" \
    --model-path "$HF_OUT" \
    --num-images 10 \
    --tp-size 1 2>&1 | tee "$HF_OUT/qualitative_eval.log"
