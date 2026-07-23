#!/bin/bash
set -eo pipefail
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
export PATH="$CARGO_HOME/bin:$PATH"
rustc --version
VENV=/workspace/sgl_venv
SGL=/workspace/torchtitan_attention_residual/sglang/python
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" -e "$SGL" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
echo "SGLANG_RETRY_DONE"
