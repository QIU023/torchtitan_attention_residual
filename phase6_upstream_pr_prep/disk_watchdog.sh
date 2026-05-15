#!/usr/bin/env bash
# disk_watchdog.sh — background poller that kill -9s the calling torchrun
# process group if disk used > ${MAX_USED_PCT:-90}%.
#
# Implements CHECKPOINT_RULES_v2.md rule 4. Launchers running > 1 hour
# MUST start this in the background:
#
#   "$WORKSPACE/phase6_upstream_pr_prep/disk_watchdog.sh" "$TORCHRUN_PGID" "$OUT_DIR" &
#   WATCHDOG_PID=$!
#   trap "kill $WATCHDOG_PID 2>/dev/null" EXIT
#
# Where TORCHRUN_PGID is the process group id of the torchrun call
# (use `setsid` or `set -m` + `kill -- -$pgid` to send to whole group).

set -u

PGID="${1:?usage: disk_watchdog.sh <torchrun-pgid> <out_dir>}"
OUT_DIR="${2:?usage: disk_watchdog.sh <torchrun-pgid> <out_dir>}"
POLL_SEC="${POLL_SEC:-60}"
MAX_USED_PCT="${MAX_USED_PCT:-90}"

mkdir -p "$OUT_DIR" 2>/dev/null
echo "$$" > "$OUT_DIR/.watchdog.pid"

echo "[watchdog] start: pgid=$PGID out=$OUT_DIR poll=${POLL_SEC}s threshold=${MAX_USED_PCT}%"

while true; do
    used=$(df --output=pcent / 2>/dev/null | tail -1 | tr -dc '0-9')
    used="${used:-0}"
    if [[ "$used" -ge "$MAX_USED_PCT" ]]; then
        echo "[watchdog] FATAL: disk used ${used}% >= ${MAX_USED_PCT}%; kill -9 -$PGID"
        df -h / 2>&1 | sed 's/^/[watchdog]   /'
        echo "[watchdog] Top consumers under workspace:"
        du -sh "$(dirname "$OUT_DIR")"/*/ 2>/dev/null | sort -hr | head -8 | sed 's/^/[watchdog]   /'

        kill -9 -- -"$PGID" 2>/dev/null
        # Also kill any DCP save subprocesses that survived the SIGKILL.
        pkill -9 -f "torchrun" 2>/dev/null || true
        echo "[watchdog] sent SIGKILL to pgid $PGID + torchrun pkill"

        # Aggressive emergency cleanup of throwaway-run ckpts so the
        # workspace can be inspected without immediate re-OOM.
        for d in "$(dirname "$OUT_DIR")"/*sweep*/checkpoint \
                 "$(dirname "$OUT_DIR")"/*smoke*/checkpoint \
                 "$(dirname "$OUT_DIR")"/*trace*/checkpoint \
                 "$(dirname "$OUT_DIR")"/*pressure*/checkpoint; do
            [[ -d "$d" ]] && rm -rf "$d" && echo "[watchdog] emergency rm $d"
        done

        rm -f "$OUT_DIR/.watchdog.pid"
        exit 1
    fi

    # Soft warning at 80%.
    if [[ "$used" -ge 80 && "$used" -lt "$MAX_USED_PCT" ]]; then
        echo "[watchdog] WARN: disk used ${used}% (threshold ${MAX_USED_PCT}%)"
    fi

    # Exit if torchrun process group is gone (run completed cleanly).
    if ! kill -0 -- -"$PGID" 2>/dev/null; then
        echo "[watchdog] torchrun pgid $PGID exited; watchdog stopping."
        rm -f "$OUT_DIR/.watchdog.pid"
        exit 0
    fi

    sleep "$POLL_SEC"
done
