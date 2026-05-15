#!/usr/bin/env bash
# Watcher: auto-publish phase7_nccl_traffic_catalog/archive at chain orchestrator milestones.
#
# Tails phase6_upstream_pr_prep/a3_v10_3d_orchestrator.log, fires publish_archive.sh
# at two known checkpoints:
#   - "running alignment report"  → A3 reached step 500
#   - "orchestrator COMPLETE"     → v10 finished
#
# Polls every 60 s. Each milestone publishes once (tracked via /tmp).

set -u
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORCH_LOG="$WORKSPACE_DIR/phase6_upstream_pr_prep/a3_v10_3d_orchestrator.log"
STATE=/tmp/phase7_publish_state
mkdir -p "$STATE"

echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) — start, polling $ORCH_LOG"

publish_once() {
    local marker="$1"; local tag="$2"
    if [[ -e "$STATE/$marker" ]]; then return 0; fi
    if grep -q "$marker" "$ORCH_LOG" 2>/dev/null; then
        echo "[watcher] $(date -u +%H:%M:%SZ) — milestone '$marker'; publishing"
        bash "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/publish_archive.sh" "$tag"
        touch "$STATE/$marker"
    fi
}

while :; do
    publish_once "running alignment report" "after A3 step 500"
    publish_once "orchestrator COMPLETE"    "after v10 5000 steps"
    if [[ -e "$STATE/orchestrator COMPLETE" ]]; then
        echo "[watcher] $(date -u +%H:%M:%SZ) — both milestones published; exit"
        exit 0
    fi
    sleep 60
done
