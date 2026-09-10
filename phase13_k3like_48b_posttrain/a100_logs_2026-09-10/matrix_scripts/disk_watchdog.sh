#!/usr/bin/env bash
# Free disk before a run dies of ENOSPC, and log exactly what it took.
#
# On 2026-08-07 the box filled to 100% mid-matrix. The failure mode is worse than
# a crashed run: the agent harness could not write a command's output file either,
# so bash stopped working entirely and nothing could be diagnosed or cleaned from
# inside. /tmp alone held 227 GB of checkpoints from matrices whose evidence is
# the 200 KB log beside them.
#
# Threshold: the request was "clean at 10 GB free". This defaults to 40 GB, and
# the reason is measured rather than cautious -- one 8-GPU 100-step cell writes
# ~700 MB of checkpoint, eighteen cells write ~12 GB, and a single seed checkpoint
# for the 2.8T-shaped flavors is larger still. At 10 GB the writer can outrun the
# sweep between two checks. Override with WATCHDOG_MIN_FREE_GB if you want 10.
#
# It only ever deletes what prune_run_artifacts.sh deletes -- checkpoint
# subdirectories and JIT caches, never logs, TensorBoard records, JSON dumps, seed
# checkpoints, or anything in a git tree. So a false alarm costs a recompile, not
# evidence.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRUNE="$HERE/prune_run_artifacts.sh"
MIN_FREE_GB=${WATCHDOG_MIN_FREE_GB:-40}
INTERVAL=${WATCHDOG_INTERVAL:-60}
ROOTS=${WATCHDOG_ROOTS:-"/tmp /workspace"}

free_gb() {
    # The mount, not the path: /tmp and /workspace are the same overlay on this
    # box, which is why cleaning /workspace alone did nothing for a full /tmp.
    df -B1G --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' '
}

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "disk watchdog: min_free=${MIN_FREE_GB}G interval=${INTERVAL}s roots=${ROOTS}"

while true; do
    avail=$(free_gb /)
    if [ -n "$avail" ] && [ "$avail" -lt "$MIN_FREE_GB" ]; then
        log "free=${avail}G below ${MIN_FREE_GB}G -- pruning"
        # shellcheck disable=SC2086
        bash "$PRUNE" $ROOTS 2>&1 | tail -20
        after=$(free_gb /)
        log "free=${after}G after pruning"
        if [ -n "$after" ] && [ "$after" -lt "$MIN_FREE_GB" ]; then
            # Say so loudly. A watchdog that runs, frees nothing, and stays quiet
            # is indistinguishable from a watchdog that is not running -- and this
            # project has lost a day to silence standing in for a measurement.
            log "WARNING: still below threshold after pruning. Nothing prunable is"
            log "WARNING: left; the space is in checkpoints, exports or datasets"
            log "WARNING: that this refuses to touch. Human decision needed."
        fi
    fi
    sleep "$INTERVAL"
done
