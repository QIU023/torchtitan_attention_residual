#!/usr/bin/env bash
# Phase 8 — qualitative VQA eval (sanity for v11 / v12 / SFT-490).
#
# Quantitative eval (lmms-eval + DCP→HF + benchmark inference) is
# 2-3 day project. For 18h budget this script does QUALITATIVE eval:
# 5 hand-picked COCO images × 5 simple VQA prompts × 3 ckpts = 75
# greedy generations side-by-side. Eyeball compares base vs SFT.
#
# Reuses phase5_vlm_multimodal_sft/generate_caption.py (already DCP-loads single-rank
# from sharded ckpt + runs greedy decode without KV cache).
#
# Outputs:
#   phase8_vqa_eval/eval_results/qual_vqa_<ckpt_label>.txt
#   phase8_vqa_eval/eval_results/qual_vqa_summary.md  (side-by-side compare)
set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/phase8_vqa_eval/eval_results"
mkdir -p "$RESULTS_DIR"

# 3 ckpts to compare
declare -A CKPTS=(
    ["v11"]="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/v11_4d_fsdp2_pp2_tp2_ep2_continue_8gpu_from_p4_step8000/checkpoint/step-5000"
    ["v12"]="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/v12_4d_fsdp2_dp2_pp2_ep2_continue_8gpu_from_p4_step8000/checkpoint/step-5000"
    ["sft490"]="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/sft_v11_llava_instruct_150k_4d/checkpoint/step-490"
)

# 5 hand-picked COCO train2017 images + prompts. Sample images from
# common categories (bus, animal, food, person, scene).
COCO_IMG_DIR=/workspace/.hf_home/coco_train2017/train2017
# SFT trained on conversation format with USER:/ASSISTANT: markers,
# so prompts must follow that pattern for the model to recognize the
# turn boundary and emit a real response (otherwise it predicts EOS
# right after the image since unmarked text is OOD for the SFT model).
PROMPTS=(
    "USER: What is in this image?"$'\n'"ASSISTANT: "
    "USER: Describe the scene briefly."$'\n'"ASSISTANT: "
    "USER: What is the main object?"$'\n'"ASSISTANT: "
    "USER: What color is the dominant subject?"$'\n'"ASSISTANT: "
    "USER: Is this an outdoor or indoor scene?"$'\n'"ASSISTANT: "
)

# 5 random COCO images. Glob expansion fails on 118K files (arg list
# too long) so use `find | shuf` instead.
mapfile -t EXISTING_IMAGES < <(
    find "$COCO_IMG_DIR" -name "*.jpg" -type f 2>/dev/null | shuf -n 5
)
if [[ ${#EXISTING_IMAGES[@]} -lt 5 ]]; then
    echo "ERROR: not enough COCO images at $COCO_IMG_DIR"; exit 1
fi
echo "Eval images:"
printf '  %s\n' "${EXISTING_IMAGES[@]}"

cd "$WORKSPACE_DIR"
for label in v11 v12 sft490; do
    ckpt="${CKPTS[$label]}"
    if [[ ! -d "$ckpt" ]]; then
        echo "[$label] missing ckpt: $ckpt — skip"
        continue
    fi
    out="$RESULTS_DIR/qual_vqa_${label}.txt"
    : > "$out"
    echo "=== $label ===" >> "$out"
    echo "ckpt: $ckpt" >> "$out"
    echo "" >> "$out"
    for i in "${!EXISTING_IMAGES[@]}"; do
        img="${EXISTING_IMAGES[$i]}"
        prompt="${PROMPTS[$i]}"
        echo "--- image $(basename "$img") | prompt: $prompt ---" >> "$out"
        timeout 180 torchrun --nproc_per_node=1 \
            phase5_vlm_multimodal_sft/generate_caption.py \
            --ckpt "$ckpt" \
            --image "$img" \
            --max-new-tokens 60 \
            --prompt "$prompt" \
            >> "$out" 2>&1 || echo "[gen failed]" >> "$out"
        echo "" >> "$out"
    done
    echo "wrote $out"
done

# Aggregate side-by-side
SUMMARY="$RESULTS_DIR/qual_vqa_summary.md"
{
    echo "# Phase 8 Qualitative VQA Eval (5 images × 5 prompts × 3 ckpts)"
    echo ""
    echo "Generated via greedy decode (no KV cache, no sampling)."
    echo "Each row is the assistant response from one ckpt."
    echo ""
    for i in "${!EXISTING_IMAGES[@]}"; do
        img="${EXISTING_IMAGES[$i]}"
        prompt="${PROMPTS[$i]}"
        echo "## $(basename "$img") — \"$prompt\""
        echo ""
        echo "| ckpt | response |"
        echo "|---|---|"
        for label in v11 v12 sft490; do
            f="$RESULTS_DIR/qual_vqa_${label}.txt"
            resp=$(awk -v hdr="--- image $(basename "$img") | prompt: $prompt ---" '
                $0 == hdr {found=1; next}
                found && /^$/ {found=0; exit}
                found {print}
            ' "$f" 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g; s/|/\\|/g')
            echo "| $label | ${resp:-(none)} |"
        done
        echo ""
    done
} > "$SUMMARY"
echo "summary written: $SUMMARY"
