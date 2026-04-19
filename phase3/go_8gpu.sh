#!/usr/bin/env bash
# Phase 3 orchestrator: get 8x RTX 5090 PCIe rental box from fresh clone
# to naive-PP and adapter-PP A/B comparison on AttnRes Llama3-150M.
#
# Prereqs (do these manually on the rental box first):
#   1. CUDA 12.x drivers, NCCL, nvidia-smi works
#   2. conda or /venv/main with Python 3.11+ and torch >= 2.11 cu12x
#   3. ssh key to github.com (for git clone of both repos)
#
# Usage:
#   bash phase3/go_8gpu.sh
#
# Env overrides:
#   WORKSPACE_DIR=/path/to/workspace
#   TORCHTITAN_DIR=/path/to/torchtitan      (default: peer of workspace)
#   HF_HOME=/mnt/ssd/hfcache                (recommended: put on big disk)
#   N_SHARDS=150                            (C4 shard prefetch count, ~45 GB)
#   NGPU=8
#   STEPS=500                               (per variant)
#   LOCAL_BS=4 GLOBAL_BS=16                 (per-device / effective batch)
#   SKIP_PREFETCH=1                         (reuse existing HF cache)
#
# Total wall time for the default 500-step smoke: ~45 min end-to-end
# once env is set up (most of it is the 2x 500-step torchrun runs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${WORKSPACE_DIR}/../torchtitan}"
N_SHARDS="${N_SHARDS:-150}"
NGPU="${NGPU:-8}"
STEPS="${STEPS:-500}"
LOCAL_BS="${LOCAL_BS:-4}"
GLOBAL_BS="${GLOBAL_BS:-16}"
SKIP_PREFETCH="${SKIP_PREFETCH:-}"

log() { echo "[go_8gpu $(date +%H:%M:%S)] $*"; }

# -- sanity --
log "workspace: ${WORKSPACE_DIR}"
log "torchtitan: ${TORCHTITAN_DIR}"
if [ ! -d "${TORCHTITAN_DIR}" ]; then
    log "ERROR: torchtitan not found at ${TORCHTITAN_DIR}"
    log "Clone it: git clone -b attention_residual_dev git@github.com:QIU023/torchtitan.git ${TORCHTITAN_DIR}"
    exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
    log "ERROR: torchrun not on PATH. Activate your python env first:"
    log "  source /venv/main/bin/activate   # or: conda activate attnres"
    exit 1
fi

log "GPU count:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -${NGPU}
if [ "$(nvidia-smi -L | wc -l)" -lt "${NGPU}" ]; then
    log "ERROR: fewer than ${NGPU} GPUs visible"
    exit 1
fi

# -- install torchtitan editable + deps (idempotent) --
log "installing torchtitan editable + deps (idempotent)"
pip install -e "${TORCHTITAN_DIR}[dev]" matplotlib tensorboard >/dev/null

# -- tokenizer --
HF_TOKENIZER_DIR="${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B"
if [ ! -f "${HF_TOKENIZER_DIR}/tokenizer.json" ]; then
    log "downloading Llama-3.1 tokenizer"
    python "${TORCHTITAN_DIR}/scripts/download_hf_assets.py" \
        --repo_id NousResearch/Meta-Llama-3.1-8B \
        --local_dir "${HF_TOKENIZER_DIR}" \
        --assets tokenizer
    # flatten nested dir if present
    if [ -d "${HF_TOKENIZER_DIR}/Meta-Llama-3.1-8B" ]; then
        mv "${HF_TOKENIZER_DIR}/Meta-Llama-3.1-8B/"* "${HF_TOKENIZER_DIR}/" || true
        rmdir "${HF_TOKENIZER_DIR}/Meta-Llama-3.1-8B"
    fi
fi

# -- prefetch C4 shards --
if [ -z "${SKIP_PREFETCH}" ]; then
    log "prefetching ${N_SHARDS} C4 shards (runs in parallel; ~45 GB for default 150)"
    python "${SCRIPT_DIR}/prefetch_c4.py" --n_shards "${N_SHARDS}"
else
    log "SKIP_PREFETCH=1, reusing existing HF cache"
fi

# -- unit tests (1 min, catches stupid bugs early) --
log "running unit tests"
( cd "${TORCHTITAN_DIR}" && python -m pytest torchtitan/experiments/attn_res/tests/ -q )

# -- naive PP run --
log "=== stage 1: naive PP, ${STEPS} steps, adapter OFF ==="
OUT_NAIVE="${SCRIPT_DIR}/runs/pp8_naive"
NGPU="${NGPU}" STEPS="${STEPS}" LOCAL_BS="${LOCAL_BS}" GLOBAL_BS="${GLOBAL_BS}" \
    OUT_DIR="${OUT_NAIVE}" TORCHTITAN_DIR="${TORCHTITAN_DIR}" \
    bash "${SCRIPT_DIR}/launch_8gpu_naive.sh"
NAIVE_RC=$?
log "naive PP exited rc=${NAIVE_RC}"

# -- adapter PP run (even if naive failed; we still want to see what adapter does) --
log "=== stage 2: PP + adapter, ${STEPS} steps, adapter ON ==="
OUT_ADAPTER="${SCRIPT_DIR}/runs/pp8_adapter"
NGPU="${NGPU}" STEPS="${STEPS}" LOCAL_BS="${LOCAL_BS}" GLOBAL_BS="${GLOBAL_BS}" \
    OUT_DIR="${OUT_ADAPTER}" TORCHTITAN_DIR="${TORCHTITAN_DIR}" \
    bash "${SCRIPT_DIR}/launch_8gpu_adapter.sh" || true
ADAPTER_RC=$?
log "adapter PP exited rc=${ADAPTER_RC}"

# -- compare --
if [ ${NAIVE_RC} -eq 0 ] && [ ${ADAPTER_RC} -eq 0 ]; then
    log "=== stage 3: compare naive vs adapter loss curves ==="
    python "${SCRIPT_DIR}/compare_pp_vs_single.py" \
        --single "${WORKSPACE_DIR}/phase2/runs/attn_res/tb" \
        --pp "${OUT_NAIVE}/tb" \
        --pp_cached "${OUT_ADAPTER}/tb" \
        || log "compare failed; inspect TB dirs manually"
else
    log "one or both runs failed; skip compare."
    log "  naive log:   ${OUT_NAIVE}/train.log"
    log "  adapter log: ${OUT_ADAPTER}/train.log"
fi

log "done. Artifacts under ${SCRIPT_DIR}/runs/"
