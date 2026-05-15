#!/usr/bin/env bash
# Pack phase7 traces + commit + push.
# Idempotent: re-runs after each completed training run produce a new
# commit with the updated tarballs (overwriting prior SNAPSHOT entries).
#
# Usage:
#   bash phase7_nccl_traffic_catalog/publish_archive.sh ["commit message tag"]
#
# Default commit subject: "phase7_nccl_traffic_catalog/archive: refresh trace tarballs"
# A custom tag is appended in parens (e.g. "after A3 step 500").

set -u
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-refresh}"
cd "$WORKSPACE_DIR"

echo "[publish] $(date -u +%Y-%m-%dT%H:%M:%SZ) — repacking…"
bash "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/pack_traces.sh" >/dev/null 2>&1 || {
    echo "[publish] pack failed; aborting"
    exit 1
}

# Stage only the archive (don't sweep up other in-flight changes)
git add phase7_nccl_traffic_catalog/archive/

if git diff --cached --quiet; then
    echo "[publish] no archive change; skip commit"
    exit 0
fi

CHANGED=$(git diff --cached --name-only | wc -l)
SIZE=$(du -sh phase7_nccl_traffic_catalog/archive | cut -f1)

git commit -m "$(cat <<EOF
phase7_nccl_traffic_catalog/archive: refresh trace tarballs ($TAG)

Idempotent re-pack via phase7_nccl_traffic_catalog/pack_traces.sh. Updated $CHANGED files,
total archive size now $SIZE. SNAPSHOT entries replaced with full
post-completion tarballs as in-flight runs finish.

Generator: phase7_nccl_traffic_catalog/publish_archive.sh
EOF
)" 2>&1 | tail -3 || {
    echo "[publish] commit failed"
    exit 1
}

echo "[publish] pushing to origin…"
git push origin main 2>&1 | tail -3
echo "[publish] DONE — new HEAD $(git log -1 --format='%h %s')"
