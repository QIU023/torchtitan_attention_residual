#!/usr/bin/env bash
# Idempotent rollback: restore upstream 0.5.0 over the live fla file.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVE=/usr/local/lib/python3.12/dist-packages/fla/modules/fused_norm_gate.py
ORIG="${DIR}/original_fla_0.5.0_fused_norm_gate.py"

if [[ ! -f "${LIVE}" || ! -f "${ORIG}" ]]; then
    echo "ERROR: missing file (live=${LIVE} or orig=${ORIG})" >&2
    exit 1
fi

live_md5=$(md5sum "${LIVE}" | awk '{print $1}')
orig_md5=$(md5sum "${ORIG}" | awk '{print $1}')

if [[ "${live_md5}" == "${orig_md5}" ]]; then
    echo "noop: already at upstream original (md5 ${live_md5})"
    exit 0
fi

echo "rolling back (live ${live_md5} → original ${orig_md5})"
cp "${ORIG}" "${LIVE}"
verify_md5=$(md5sum "${LIVE}" | awk '{print $1}')
if [[ "${verify_md5}" != "${orig_md5}" ]]; then
    echo "ERROR: post-rollback md5 mismatch (got ${verify_md5}, expected ${orig_md5})" >&2
    exit 3
fi
echo "rolled back OK"
