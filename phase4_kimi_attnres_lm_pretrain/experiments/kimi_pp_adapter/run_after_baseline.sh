#!/usr/bin/env bash
# Auto-chain Problem B (PP-adapter run) after the active Problem A
# AttnRes FSDP run finishes.
#
# Polls phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp_overnight/train.log
# for "Training completed", then launches the single adapter_pp arm
# at STEPS=12500 — matched to Problem A's 436M AttnRes FSDP run, so
# the loss curves are directly comparable across the two
# parallelism strategies (FSDP vs PP+adapter on the same model,
# same effective batch, same total tokens).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTNRES_LOG="${SCRIPT_DIR}/../../runs/kimi_436m_block_attn_res_fsdp_overnight/train.log"

echo "[$(date -Is)] Waiting for AttnRes FSDP run to finish: ${ATTNRES_LOG}"
until grep -qE "Training completed|Process group destroyed" "${ATTNRES_LOG}" 2>/dev/null; do
    sleep 60
done
echo "[$(date -Is)] AttnRes FSDP run done. GPU drain wait..."
sleep 30

echo "[$(date -Is)] === Starting Problem B: adapter_pp (12500 steps) ==="
bash "${SCRIPT_DIR}/launch_adapter_pp.sh"
echo "[$(date -Is)] === adapter_pp exited rc=$? ==="
