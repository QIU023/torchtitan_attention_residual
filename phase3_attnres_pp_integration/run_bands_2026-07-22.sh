#!/usr/bin/env bash
# naive-vs-naive band for pp8vp2 and pp4vp4 (pp4vp2 band already measured),
# to contextualize every adapter |dLoss| against its own config's noise floor.
set -uo pipefail
P3=/workspace/torchtitan_attention_residual/phase3_attnres_pp_integration
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" HF_HOME=/workspace/.hf_home
for spec in "pp8vp2:175m_attn_res_L16_n8:8:2:16:16" "pp4vp4:175m_attn_res_L16_n8:4:4:16:32"; do
  tag="${spec%%:*}"; sweep="${spec#*:}"
  STEPS=1000 NGPU=8 RUN_ADAPTER=0 SWEEP="$sweep" \
    SWEEP_OUT_ROOT="$P3/runs/band_${tag}_naive2_20260722" \
    bash "$P3/run_pp_pressure_test.sh" >>"$P3/band_${tag}.log" 2>&1 || true
done
echo "BANDS_DONE" >> "$P3/OVERNIGHT_RESULTS_2026-07-22.md"
