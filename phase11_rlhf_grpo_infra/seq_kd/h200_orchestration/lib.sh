#!/usr/bin/env bash
# Shared helpers for the overnight seq-KD pipeline.
ROOT=/home/seqkd_overnight
REPO=/home/torchtitan_attention_residual
CPY=/root/miniconda3/envs/py3.10/bin/python
CTORCHRUN=/root/miniconda3/envs/py3.10/bin/torchrun
VPY=/home/venv/vllm/bin/python
export HF_HOME=/home/.hf_home
# Python.h missing in conda env -> torch.compile inductor C++ build fails.
# Disable dynamo so torch.compile is a no-op (eager). Fine for 447M SFT.
export TORCHDYNAMO_DISABLE=1
STATUS="${ROOT}/STATUS"
mkdir -p "${ROOT}/logs"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }
status() { echo "[$(ts)] $*" >> "${STATUS}"; }

disk_free_g() { df -BG --output=avail /home | tail -1 | tr -dc 0-9; }

# preflight: require >= N GB free, else return 1
require_disk() {
    local need="${1:-30}" f
    f=$(disk_free_g)
    if (( f < need )); then
        log "DISK preflight FAIL: ${f}G < ${need}G"
        return 1
    fi
    log "DISK ok: ${f}G free (need ${need}G)"
    return 0
}

# stage output checkpoint exists? (dir with .metadata)
ckpt_ok() { [[ -f "$1/.metadata" ]]; }

latest_step_ckpt() { ls -d "$1"/step-* 2>/dev/null | sort -t- -k2 -n | tail -1; }
