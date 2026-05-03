#!/usr/bin/env bash
# Restore the venv's stage.py to the vendored original.
set -eu
PATCHES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/stage.py
cp "$PATCHES_DIR/stage.py.original" "$TARGET"
echo "[restore] reverted $TARGET to vendored original"
