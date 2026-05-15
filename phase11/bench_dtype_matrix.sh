#!/usr/bin/env bash
# Run the inference dtype matrix (A/B/C) sequentially. Each row gets its own
# Engine process; results stream to a per-config log + a summary file.
#
#   A: bf16   + torch_native decode (current baseline; coherent path)
#   B: fp16   + torch_native decode (fp16 dynamic-range stress-test)
#   C: bf16   + fp8 weight-only quant + torch_native decode (weight-quant)
set -uo pipefail   # NOTE: no -e — keep going if one config fails (we want all 3 results).
cd /workspace/torchtitan_attention_residual

MODEL=${MODEL:-${PWD}/phase5/runs/mm_sft_447m_full/hf_step3100}
N=${N:-8}
TOKS=${TOKS:-64}
OUT=${OUT:-/tmp/bench_dtype_matrix_$(date -u +%Y%m%dT%H%M%S)}
mkdir -p "${OUT}"

run() {
    local tag="$1"; shift
    local log="${OUT}/${tag}.log"
    echo
    echo "================================================================"
    echo "[matrix] === ${tag} ==="
    echo "================================================================"
    /usr/bin/python3 phase11/bench_inference_dtype.py \
        --model-path "${MODEL}" \
        --num-samples "${N}" \
        --max-new-tokens "${TOKS}" \
        "$@" 2>&1 | tee "${log}" || echo "[matrix] ${tag} FAILED (continuing)"
}

run A_bf16_torchnative \
    --dtype bfloat16 \
    --decode-attention-backend torch_native

run B_fp16_torchnative \
    --dtype float16 \
    --decode-attention-backend torch_native

run C_fp8wq_torchnative \
    --dtype bfloat16 \
    --quantization fp8 \
    --decode-attention-backend torch_native

echo
echo "================================================================"
echo "[matrix] SUMMARY"
echo "================================================================"
for f in "${OUT}"/*.log; do
    echo "--- $(basename ${f}) ---"
    grep -E "BOOT_S|TOTAL_S|AVG_S|TOTAL_TOKENS|TOKENS_PER_S|COHERENT" "${f}" | sed 's/.*\[bench\] //'
done | tee "${OUT}/SUMMARY.txt"
echo
echo "[matrix] artifacts at ${OUT}"
