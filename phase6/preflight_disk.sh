#!/usr/bin/env bash
# preflight_disk.sh — disk discipline preflight, sourced at the top of every
# launcher. Implements CHECKPOINT_RULES_v2.md rules 2, 3, 5.
#
# Effects (in order):
#   1. If OUT_DIR matches a "throwaway" pattern (sweep / smoke / align /
#      trace / pressure / naive / adapter), force CHECKPOINT_ENABLED=0
#      regardless of caller setting.
#   2. Auto-prune stale checkpoint dirs from prior throwaway runs to
#      free disk.
#   3. df -h check; refuse to launch if free < ${MIN_FREE_GB:-80} GB.
#
# Source (do NOT exec):
#     source "$(dirname "${BASH_SOURCE[0]}")/preflight_disk.sh"

set -u

MIN_FREE_GB="${MIN_FREE_GB:-80}"
WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-$WORKSPACE/phase5/runs}"

if [[ -z "${OUT_DIR:-}" ]]; then
    echo "[preflight] WARN: OUT_DIR not set; pattern check skipped"
else
    case "$OUT_DIR" in
        *sweep*|*pressure*|*smoke*|*align*|*trace*|*tier_a*|*tier_b*|*tier_c*|*tier_d*|*_naive_*|*_adapter_*)
            if [[ "${CHECKPOINT_ENABLED:-0}" != "0" ]]; then
                echo "[preflight] OUT_DIR='$OUT_DIR' matches a throwaway pattern."
                echo "[preflight] Forcing CHECKPOINT_ENABLED=0 (was ${CHECKPOINT_ENABLED:-unset})."
                export CHECKPOINT_ENABLED=0
            fi
            ;;
    esac
fi

# Auto-prune stale ckpts from prior throwaway runs.
if [[ -d "$RUNS_ROOT" ]]; then
    pruned=0
    while IFS= read -r d; do
        case "$d" in
            *sweep*|*pressure*|*smoke*|*align*|*trace*|*tier_a*|*tier_b*|*tier_c*|*tier_d*|*_naive_*|*_adapter_*)
                if [[ -d "$d/checkpoint" ]]; then
                    sz=$(du -sh "$d/checkpoint" 2>/dev/null | awk '{print $1}')
                    echo "[preflight] auto-prune $d/checkpoint (${sz})"
                    rm -rf "$d/checkpoint"
                    pruned=$((pruned + 1))
                fi
                ;;
        esac
    done < <(find "$RUNS_ROOT" -maxdepth 2 -mindepth 1 -type d 2>/dev/null)
    if [[ "$pruned" -gt 0 ]]; then
        echo "[preflight] pruned $pruned throwaway-run ckpt dir(s)"
    fi
fi

# Pre-launch df check.
free_gb=$(df -BG / 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
free_gb="${free_gb:-0}"
if [[ "$free_gb" -lt "$MIN_FREE_GB" ]]; then
    echo "[preflight] FATAL: only ${free_gb}GB free, need ${MIN_FREE_GB}GB."
    echo "[preflight]   df -h / output:"
    df -h / 2>&1 | sed 's/^/    /'
    echo "[preflight]   Top space consumers under $RUNS_ROOT:"
    du -sh "$RUNS_ROOT"/*/ 2>/dev/null | sort -hr | head -10 | sed 's/^/    /'
    echo "[preflight]   Free disk manually before re-launching."
    exit 1
fi

echo "[preflight] disk OK: ${free_gb}GB free (>= ${MIN_FREE_GB}GB required)."
echo "[preflight] CHECKPOINT_ENABLED=${CHECKPOINT_ENABLED:-unset}"
echo "[preflight] KEEP_K=${KEEP_K:-1} SAVE_FREQ=${SAVE_FREQ:-1000}"
