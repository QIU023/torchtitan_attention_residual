#!/usr/bin/env bash
# Download Priority B + MMBench retry. Disk-aware (abort if free <25G).
set -euo pipefail
ulimit -c 0

EVAL_DIR="${EVAL_DIR:-/workspace/.hf_home/eval_data}"
mkdir -p "${EVAL_DIR}"
PY=/usr/bin/python3

disk_check() {
    local free=$(df -BG /workspace | awk 'NR==2{gsub("G","",$4); print $4}')
    if (( free < 25 )); then echo "[$(date)] ABORT disk ${free}G < 25G" >&2; exit 1; fi
    echo "[$(date)] disk OK: ${free}G free"
}

# ---------- MMBench (HF mirror) ----------
echo "[$(date)] === MMBench retry from HF ==="
disk_check
MB="${EVAL_DIR}/mmbench"
mkdir -p "${MB}"
${PY} -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lmms-lab/MMBench', repo_type='dataset',
                  local_dir='${MB}', local_dir_use_symlinks=False,
                  allow_patterns=['*.tsv','*.json','*.parquet','*.csv'])
" 2>&1 | tail -5
echo "  ✓ mmbench (lmms-lab mirror)"

# ---------- VQAv2 ----------
echo "[$(date)] === VQAv2 test-dev annotations ==="
disk_check
VQA="${EVAL_DIR}/vqav2"
mkdir -p "${VQA}"
for url in \
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" \
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" \
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip" ; do
    fname=$(basename "$url")
    if [[ ! -f "${VQA}/${fname%.zip}.json" && ! -f "${VQA}/${fname}" ]]; then
        curl -fsSL -o "${VQA}/${fname}" "$url"
        ( cd "${VQA}" && unzip -qo "${fname}" && rm "${fname}" )
    fi
done
echo "  ✓ VQAv2 annotations (use POPE's COCO val2014 images; test2015 = $((5))GB extra optional)"

# ---------- GQA ----------
echo "[$(date)] === GQA test-dev-balanced ==="
disk_check
GQA="${EVAL_DIR}/gqa"
mkdir -p "${GQA}"
${PY} -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lmms-lab/GQA', repo_type='dataset',
                  local_dir='${GQA}', local_dir_use_symlinks=False,
                  allow_patterns=['*test*','*val*','*.json','*.parquet','images/**'])
" 2>&1 | tail -5
echo "  ✓ GQA"

echo "[$(date)] === Priority B done ==="
df -BG /workspace | tail -1
du -sh "${EVAL_DIR}"/* 2>/dev/null
