#!/usr/bin/env bash
# Disk panic watchdog.
#
# Polls /workspace free space every 60s. If free drops below ${PANIC_GB}
# (default 12G), immediately SIGTERMs all training/orchestrator processes
# to prevent disk-full → user-locked-out box.
#
# Why this is independent of the orchestrator's check_disk():
#   - check_disk() runs only at stage boundaries (~once per hour during
#     stage 2). Between boundaries, ckpt writes can fill disk rapidly.
#   - The orchestrator could be wedged in a syscall (e.g. trim_ckpts
#     during a busy I/O period) when disk fills.
#   - A separate process gives us a true safety net the orchestrator
#     can't accidentally disable.
#
# Targets killed when panicking:
#   - run_overnight_pipeline.sh (orchestrator)
#   - torchrun -m torchtitan.train ... (any stage)
#   - python -m phase5_vlm_multimodal_sft.train_mm (multimodal SFT)
#   - run_grpo_*.py (GRPO entrypoints)
#
# Logs to /workspace/.disk_panic_watchdog.log so user can see what happened.

set -uo pipefail

PANIC_GB="${PANIC_GB:-12}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_FILE="${LOG_FILE:-/workspace/.disk_panic_watchdog.log}"

exec >>"${LOG_FILE}" 2>&1

free_gb() { df -BG /workspace | awk 'NR==2{gsub("G","",$4);print $4}'; }
now() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(now)] watchdog started: PANIC_GB=${PANIC_GB} POLL=${POLL_SECONDS}s"

panic() {
    local free="$1"
    echo "[$(now)] ============================================================"
    echo "[$(now)] DISK PANIC — free=${free}G < ${PANIC_GB}G threshold"
    echo "[$(now)] sending SIGTERM to all training+orchestrator processes"
    echo "[$(now)] ============================================================"

    # Order: orchestrator first (prevent new stages), then training, then sglang/grpo
    pkill -TERM -f 'run_overnight_pipeline.sh' 2>/dev/null && echo "  killed orchestrator"
    pkill -TERM -f 'torchrun' 2>/dev/null && echo "  killed torchrun"
    pkill -TERM -f 'torchtitan.train' 2>/dev/null && echo "  killed torchtitan.train"
    pkill -TERM -f 'train_mm' 2>/dev/null && echo "  killed train_mm"
    pkill -TERM -f 'run_grpo_' 2>/dev/null && echo "  killed grpo"
    pkill -TERM -f 'sglang.launch_server' 2>/dev/null && echo "  killed sglang server"

    # 30s grace, then SIGKILL stragglers
    sleep 30
    pkill -KILL -f 'torchrun' 2>/dev/null || true
    pkill -KILL -f 'torchtitan.train' 2>/dev/null || true
    pkill -KILL -f 'train_mm' 2>/dev/null || true
    pkill -KILL -f 'run_grpo_' 2>/dev/null || true

    # NCCL/torch crashes drop 8-12GB core dumps to /var/lib/vastai_kaalia/data/.
    # 8 ranks × repeated KDA crashes = 100GB+ accumulation, which itself fills the
    # disk. We've never opened these in gdb (Python tracebacks in stage logs are
    # enough), so clear them on panic to free the most space possible.
    if [[ -d /var/lib/vastai_kaalia/data ]]; then
        local before=$(du -sb /var/lib/vastai_kaalia/data/core-* 2>/dev/null | awk '{s+=$1} END {printf "%.0fG", s/1024/1024/1024}')
        rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null
        echo "  cleared core dumps from /var/lib/vastai_kaalia/data (was ~${before:-0G})"
    fi

    sleep 5
    local final=$(free_gb)
    echo "[$(now)] PANIC complete. free now=${final}G"
    echo "[$(now)] watchdog exiting (one-shot trip)"
    exit 0
}

while true; do
    local_free=$(free_gb)
    if (( local_free < PANIC_GB )); then
        panic "${local_free}"
    fi
    sleep "${POLL_SECONDS}"
done
