#!/usr/bin/env bash
# Overlay the patched stage.py into the venv's pytorch installation.
# Pairs with restore.sh which puts the original back.
#
# The patch fixes the PP+Interleaved1F1B+V≥2+LBS≥2 backward graph
# reuse bug. See:
#   additional_found_issues/torchtitan_pp_microbatch_backward_graph.md
#   additional_found_issues/torchtitan_pp_lbs_backward_INVESTIGATION.md
set -eu

PATCHES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/stage.py

# Sanity: original on disk should match our vendored snapshot.
if ! diff -q "$PATCHES_DIR/stage.py.original" "$TARGET" >/dev/null 2>&1; then
    if [[ "${FORCE:-0}" != "1" ]]; then
        echo "ERROR: $TARGET differs from $PATCHES_DIR/stage.py.original"
        echo "Either:"
        echo "  - venv was already patched (run restore.sh first)"
        echo "  - or pytorch was upgraded (re-vendor stage.py.original)"
        echo "Force overwrite anyway: FORCE=1 bash $0"
        exit 1
    fi
fi

cp "$PATCHES_DIR/stage.py.patched" "$TARGET"
echo "[apply] patched $TARGET"
echo "[apply] verify: python -c \"from torch.distributed.pipelining.stage import _PipelineStageBase; print('ok')\""
