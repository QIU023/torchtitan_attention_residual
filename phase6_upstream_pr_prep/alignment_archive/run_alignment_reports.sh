#!/usr/bin/env bash
# Phase 6 alignment report generator.
#
# Once all 8-GPU alignment runs are done, run compare_pp_vs_fsdp.py
# against the B0 anchor for each config and emit a per-config report
# under phase6_upstream_pr_prep/alignment_<config>.{txt,csv,png}.

set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE6="$WORKSPACE_DIR/phase6"
# Each train.log dump produces tb/<timestamp>/events.out.tfevents.* —
# pick the most recent timestamp dir per run.
_resolve_tb() {
    local run_dir="$1"
    local tb_root="$run_dir/tb"
    [[ -d "$tb_root" ]] || return 1
    ls -dt "$tb_root"/*/ 2>/dev/null | head -1 | sed 's:/$::'
}

ANCHOR_TB="$(_resolve_tb "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_b0_fsdp8_seed42")"
if [[ -z "$ANCHOR_TB" || ! -d "$ANCHOR_TB" ]]; then
    echo "ERROR: B0 anchor TB not found under phase5_vlm_multimodal_sft/runs/8gpu_b0_fsdp8_seed42/tb/" >&2
    exit 1
fi
echo "[anchor] $ANCHOR_TB"

cd "$WORKSPACE_DIR"

report() {
    local cfg="$1"
    local tb
    tb="$(_resolve_tb "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_${cfg}_seed42")"
    local out_txt="$PHASE6/alignment_8gpu_${cfg}.txt"
    local out_csv="$PHASE6/alignment_8gpu_${cfg}.csv"
    local out_png="$PHASE6/alignment_8gpu_${cfg}.png"
    if [[ -z "$tb" || ! -d "$tb" ]]; then
        echo "[skip] $cfg: TB not found"
        return 0
    fi
    /venv/main/bin/python phase5_vlm_multimodal_sft/compare_pp_vs_fsdp.py \
        --pp "$tb" \
        --fsdp "$ANCHOR_TB" \
        --noise-band 0.13 \
        --out-csv "$out_csv" \
        --out-plot "$out_png" \
        > "$out_txt" 2>&1 || true
    echo "[done] $cfg → $out_txt"
    head -25 "$out_txt"
    echo "---"
}

report a2_fsdp2_pp4
report a3_fsdp2_pp2_tp2
report a6_fsdp4_pp2_ep2
report all4d_fsdp2_pp2_tp2_ep2
report noppc_fsdp4_tp2_ep2
