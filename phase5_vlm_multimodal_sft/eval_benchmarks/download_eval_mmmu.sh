#!/usr/bin/env bash
# Download MMMU validation split (LLaVA-1.5 paper eval). Disk-aware.
set -euo pipefail
ulimit -c 0

EVAL_DIR="${EVAL_DIR:-/workspace/.hf_home/eval_data}"
MMMU="${EVAL_DIR}/mmmu"
mkdir -p "${MMMU}"
PY=/usr/bin/python3

disk_check() {
    local free=$(df -BG /workspace | awk 'NR==2{gsub("G","",$4); print $4}')
    if (( free < 25 )); then echo "[$(date)] ABORT disk ${free}G < 25G" >&2; exit 1; fi
    echo "[$(date)] disk OK: ${free}G free"
}

echo "[$(date)] === MMMU validation split (lmms-lab mirror) ==="
disk_check
# lmms-lab/MMMU has validation + test; we use dev+val (test is held-out)
${PY} -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lmms-lab/MMMU', repo_type='dataset',
                  local_dir='${MMMU}', local_dir_use_symlinks=False,
                  allow_patterns=['*validation*','*dev*','*.json','*.parquet'])
" 2>&1 | tail -8
echo "[$(date)] === MMMU done ==="
du -sh "${MMMU}" 2>/dev/null
df -BG /workspace | tail -1
