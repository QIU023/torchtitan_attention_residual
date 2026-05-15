#!/usr/bin/env bash
# Phase 2 environment setup on a fresh Linux box (tested: Ubuntu 22.04, CUDA 12.4+).
#
# Creates a conda env named `attnres`, installs torch nightly + torchtitan
# dependencies, downloads the Llama-3.1-8B tokenizer, and verifies the
# install by running the standalone AttnRes smoke test and unit tests.
#
# Expected runtime: ~10-15 minutes (mostly pip/conda downloads).
# Disk footprint: ~15 GB for env + ~1 GB for tokenizer.
#
# Assumed layout (phase2_attnres_baseline_loss/ is a peer of torchtitan/, not inside it):
#   <workspace>/
#   ├── phase2_attnres_baseline_loss/          <- this script lives here
#   └── torchtitan/      <- cloned fork, feat/block-attn-res branch
#
# Prereqs:
#   - conda (miniconda/anaconda) already installed
#   - torchtitan fork already cloned at ../torchtitan on feat/block-attn-res
#   - HuggingFace auth (defaults to NousResearch mirror, no login required)
#
# Usage (from workspace root):
#   bash phase2_attnres_baseline_loss/setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"
ENV_NAME="${ENV_NAME:-attnres}"
PY_VERSION="${PY_VERSION:-3.11}"
# Use the NousResearch mirror by default (ungated, identical tokenizer to
# meta-llama/Llama-3.1-8B). Override with HF_REPO=meta-llama/Llama-3.1-8B
# if you prefer the official source (requires license acceptance).
HF_REPO="${HF_REPO:-NousResearch/Meta-Llama-3.1-8B}"
HF_LOCAL_DIR="${HF_LOCAL_DIR:-${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B}"

if [ ! -d "${TORCHTITAN_DIR}" ]; then
    echo "[setup_env] TORCHTITAN_DIR not found: ${TORCHTITAN_DIR}"
    echo "[setup_env] Clone your fork first:"
    echo "    git clone -b feat/block-attn-res https://github.com/QIU023/torchtitan.git ${TORCHTITAN_DIR}"
    exit 1
fi

echo "[setup_env] Creating conda env: ${ENV_NAME} (python ${PY_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[setup_env] Installing torch nightly (CUDA 12.4 build)"
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

echo "[setup_env] Installing torchtitan editable + dev deps"
pushd "${TORCHTITAN_DIR}" >/dev/null
pip install -e ".[dev]"
popd >/dev/null

echo "[setup_env] Installing extra runbook deps (matplotlib, tensorboard)"
pip install matplotlib tensorboard

echo "[setup_env] Verifying torch CUDA visibility"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "[setup_env] Downloading tokenizer from ${HF_REPO} to ${HF_LOCAL_DIR}"
pushd "${TORCHTITAN_DIR}" >/dev/null
python scripts/download_hf_assets.py \
    --repo_id "${HF_REPO}" \
    --local_dir "${HF_LOCAL_DIR}" \
    --asset_types tokenizer
popd >/dev/null

echo "[setup_env] Smoke test: standalone AttnRes primitive"
python "${SCRIPT_DIR}/smoke_test_attn_res.py"

echo "[setup_env] Smoke test: torchtitan unit tests for AttnRes"
pushd "${TORCHTITAN_DIR}" >/dev/null
python -m pytest tests/unit_tests/test_attn_res.py -v
popd >/dev/null

echo "[setup_env] DONE. Activate with: conda activate ${ENV_NAME}"
