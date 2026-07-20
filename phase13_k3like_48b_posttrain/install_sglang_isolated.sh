#!/bin/bash
# Isolated sglang env for the AttnRes rollout server (torch 2.11 pin).
# Separate process from veRL (/venv/verl torch 2.12); they talk over HTTP,
# so the torch-version conflict (sglang 2.11 vs titan 2.12) is sidestepped.
set -eo pipefail
export UV_HTTP_TIMEOUT=300
VENV=/workspace/sgl_venv
SGL=/workspace/torchtitan_attention_residual/sglang/python
uv venv --python 3.12 "$VENV"
# torch 2.11 cu128 (Blackwell sm_120; driver CUDA 13.0 accepts it)
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
  "torch==2.11.0" torchvision \
  --index-url https://download.pytorch.org/whl/cu128
# sglang fork (prebuilt flashinfer/sgl-kernel wheels, no CUDA compile)
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" -e "$SGL" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
echo "SGLANG_INSTALL_DONE"
# --- post-resolve fixups (uv's unsafe-best-match re-picks these wrong) ---
# 1) sglang re-resolves torch to +cu130; realign torchvision to match.
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu130 \
  "torchvision==0.26.0+cu130" --no-deps --reinstall
# 2) sglang leaves kernels unpinned -> pulls 0.16 (too new); transformers
#    5.6.0 needs kernels>=0.12,<0.13 (LayerRepository version-arg API).
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
  "kernels>=0.12.0,<0.13" --no-deps --reinstall
echo "SGLANG_FIXUPS_DONE"
# Build needs: rustup (Rust) + protoc on PATH for the sglang-grpc crate.
