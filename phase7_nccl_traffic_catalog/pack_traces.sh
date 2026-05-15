#!/usr/bin/env bash
# Pack phase7 NCCL traces for cluster traffic replay.
#
# Each run dir is packed as a self-contained tarball containing:
#   - tier_*_trace/nccl-rank-*.log  (raw NCCL_DEBUG=INFO logs)
#   - tier_*_trace/collective_summary.csv  (parsed histogram)
#   - tier_*_trace/nsys-* if any
#   - recipe.json (mesh + recipe)
#   - train.log (loss curve provenance)
#   - tb/  (TensorBoard event files)
#
# Usage:
#   bash phase7_nccl_traffic_catalog/pack_traces.sh [output_dir]
#
# Default output_dir: /workspace/phase7_archive

set -u
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$WORKSPACE_DIR/phase7_nccl_traffic_catalog/archive}"
mkdir -p "$OUT"

MANIFEST="$OUT/MANIFEST.md"
echo "# Phase 7 NCCL trace archive" > "$MANIFEST"
echo "" >> "$MANIFEST"
echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
echo "Hostname: $(hostname)" >> "$MANIFEST"
echo "Hardware: 8× RTX 5090 PCIe (Blackwell sm_120, 32 GiB / GPU)" >> "$MANIFEST"
echo "Workspace HEAD: $(cd "$WORKSPACE_DIR" && git log -1 --format="%H %s")" >> "$MANIFEST"
echo "Submodule HEAD: $(cd "$WORKSPACE_DIR/torchtitan" && git log -1 --format="%H %s")" >> "$MANIFEST"
echo "" >> "$MANIFEST"
echo "Each per-run tarball is self-contained. Untar and replay with" >> "$MANIFEST"
echo "phase7_nccl_traffic_catalog/extract_collectives.py (re-emit CSV from raw NCCL log)." >> "$MANIFEST"
echo "" >> "$MANIFEST"
echo "## Archives" >> "$MANIFEST"
echo "" >> "$MANIFEST"
echo "| Archive | Mesh | Tier | GBS | Steps | Size (gz) | sha256 |" >> "$MANIFEST"
echo "|---|---|---|---|---|---|---|" >> "$MANIFEST"

pack_one() {
    local run_dir="$1"
    local archive_name="$2"
    local mesh="$3"
    local tier="$4"
    local gbs="$5"
    local steps_target="$6"

    if [[ ! -d "$run_dir" ]]; then
        echo "[skip] $run_dir does not exist"
        return 0
    fi

    # Trace dir(s) inside the run dir
    local has_trace=0
    for td in "$run_dir"/tier_a_trace "$run_dir"/tier_b_trace "$run_dir"/tier_c_trace; do
        [[ -d "$td" && -n "$(ls "$td"/nccl-rank-*.log 2>/dev/null)" ]] && has_trace=1
    done
    if [[ $has_trace == 0 ]]; then
        echo "[skip] $run_dir has no NCCL log files"
        return 0
    fi

    local tarball="$OUT/${archive_name}.tar.gz"
    echo "[pack] $run_dir → $tarball"

    # Tar only the contents we want; exclude any checkpoint dirs / comm_traces
    cd "$WORKSPACE_DIR"
    tar -czf "$tarball" \
        --exclude="checkpoint" \
        --exclude="comm_traces" \
        --exclude="*.distcp" \
        --exclude=".metadata" \
        "$run_dir"/recipe.json \
        "$run_dir"/train.log \
        "$run_dir"/tb \
        "$run_dir"/tier_a_trace \
        "$run_dir"/tier_b_trace \
        "$run_dir"/tier_c_trace \
        2>/dev/null || true

    if [[ ! -s "$tarball" ]]; then
        echo "[err] $tarball is empty"
        rm -f "$tarball"
        return 1
    fi

    local sz=$(du -sh "$tarball" | cut -f1)
    local sha=$(sha256sum "$tarball" | cut -c1-12)
    echo "  size=$sz  sha256=$sha"
    echo "| \`$(basename "$tarball")\` | $mesh | $tier | $gbs | $steps_target | $sz | \`${sha}…\` |" >> "$MANIFEST"
}

# ===== completed runs =====
pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_b0_fsdp8_seed42"        "b0_fsdp8_alignment"     "FSDP=8 PP=1"            "tier_c" "16"  "500"
pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_b0_fsdp8_tier_b"        "b0_fsdp8_tier_b"        "FSDP=8 PP=1"            "tier_b" "120" "50"
pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_b0_fsdp8_tier_a"        "b0_fsdp8_tier_a"        "FSDP=8 PP=1"            "tier_a" "384" "100"
pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_a2_fsdp2_pp4_seed42"    "a2_fsdp2_pp4_alignment" "FSDP=2 PP=4 V=2"         "tier_c" "16"  "500"
pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_a2_fsdp2_pp4_tier_b"    "a2_fsdp2_pp4_tier_b"    "FSDP=2 PP=4 V=2"         "tier_b" "120" "50 (failed)"
pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_a2_fsdp2_pp4_tier_a"    "a2_fsdp2_pp4_tier_a"    "FSDP=2 PP=4 V=2"         "tier_a" "384" "100 (failed)"

# ===== in-flight (snapshot only, will repack later) =====
if [[ -d "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_a3_fsdp2_pp2_tp2_seed42" ]]; then
    pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_a3_fsdp2_pp2_tp2_seed42" "a3_fsdp2_pp2_tp2_alignment_SNAPSHOT" "FSDP=2 PP=2 TP=2 V=2" "tier_c" "16" "500 (in flight)"
fi
if [[ -d "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/v10_continue_8gpu_from_p4_step8000" ]]; then
    pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/v10_continue_8gpu_from_p4_step8000" "v10_fsdp8_pretrain_PARTIAL" "FSDP=8 PP=1" "tier_b" "120" "interrupted"
fi
if [[ -d "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/v10_3d_continue_8gpu_from_p4_step8000" ]]; then
    pack_one "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/v10_3d_continue_8gpu_from_p4_step8000" "v10_3d_fsdp2pp2tp2_pretrain_SNAPSHOT" "FSDP=2 PP=2 TP=2 V=2" "tier_b" "120" "5000 (3D)"
fi

# ===== bundle the catalog + extract scripts =====
echo "" >> "$MANIFEST"
echo "## Helper scripts (in tools.tar.gz)" >> "$MANIFEST"
echo "" >> "$MANIFEST"
echo "- \`phase7_nccl_traffic_catalog/extract_collectives.py\` — parses NCCL_DEBUG=INFO logs into structured CSV" >> "$MANIFEST"
echo "- \`phase7_nccl_traffic_catalog/build_pattern_catalog.py\` — aggregates per-run CSVs into pattern_catalog.md" >> "$MANIFEST"
echo "- \`phase7_nccl_traffic_catalog/pattern_catalog.md\` — human-readable cross-config histogram (current snapshot)" >> "$MANIFEST"
echo "- \`phase7_nccl_traffic_catalog/README.md\` — three-tier recording rationale" >> "$MANIFEST"

cd "$WORKSPACE_DIR"
tar -czf "$OUT/tools.tar.gz" \
    phase7_nccl_traffic_catalog/extract_collectives.py \
    phase7_nccl_traffic_catalog/build_pattern_catalog.py \
    phase7_nccl_traffic_catalog/pattern_catalog.md \
    phase7_nccl_traffic_catalog/README.md \
    phase6_upstream_pr_prep/SESSION_8GPU_summary.md \
    2>/dev/null
tools_sz=$(du -sh "$OUT/tools.tar.gz" | cut -f1)
echo "[pack] tools.tar.gz  size=$tools_sz"

# ===== summary =====
echo ""
echo "=================================="
echo "Total archive size: $(du -sh "$OUT" | cut -f1)"
echo "Manifest:           $MANIFEST"
echo "=================================="
ls -lh "$OUT"
