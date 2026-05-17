#!/usr/bin/env bash
# Idempotent apply: copy patched_fused_norm_gate.py over the live fla file.
# Refuses to run if the current live file doesn't match either the original
# backup or our patched copy (i.e. someone else changed it — investigate first).

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVE=/usr/local/lib/python3.12/dist-packages/fla/modules/fused_norm_gate.py
ORIG="${DIR}/original_fla_0.5.0_fused_norm_gate.py"
PATCHED="${DIR}/patched_fused_norm_gate.py"

if [[ ! -f "${LIVE}" ]]; then
    echo "ERROR: live fla file missing: ${LIVE}" >&2
    exit 1
fi
if [[ ! -f "${ORIG}" || ! -f "${PATCHED}" ]]; then
    echo "ERROR: backup or patched copy missing under ${DIR}" >&2
    exit 1
fi

live_md5=$(md5sum "${LIVE}" | awk '{print $1}')
orig_md5=$(md5sum "${ORIG}" | awk '{print $1}')
patched_md5=$(md5sum "${PATCHED}" | awk '{print $1}')

if [[ "${live_md5}" == "${patched_md5}" ]]; then
    echo "noop: patch already applied (md5 ${live_md5})"
    exit 0
fi

if [[ "${live_md5}" != "${orig_md5}" ]]; then
    echo "ERROR: live file md5=${live_md5} matches neither original (${orig_md5}) nor patched (${patched_md5})." >&2
    echo "Refusing to overwrite — someone else (pip upgrade? another patcher?) changed it." >&2
    echo "Investigate manually before re-running apply.sh." >&2
    exit 2
fi

echo "applying patch (live ${live_md5} → patched ${patched_md5})"
cp "${PATCHED}" "${LIVE}"
verify_md5=$(md5sum "${LIVE}" | awk '{print $1}')
if [[ "${verify_md5}" != "${patched_md5}" ]]; then
    echo "ERROR: post-apply md5 mismatch (got ${verify_md5}, expected ${patched_md5})" >&2
    exit 3
fi
echo "applied OK"
