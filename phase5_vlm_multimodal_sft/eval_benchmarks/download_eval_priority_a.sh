#!/usr/bin/env bash
# Download Priority A eval benchmarks for LLaVA-style downstream eval.
# Total: ~10-15GB. Disk-aware: aborts if /workspace free < 30G.
#
# Sources (all standard LLaVA-1.5 paper eval set):
#   1. TextVQA val      → AllenAI / LLaVA mirror
#   2. POPE             → AGI-Edgerunners/POPE + COCO val2014
#   3. MM-Vet           → yuweihao/MM-Vet HF
#   4. LLaVA-Bench-Wild → liuhaotian/llava-bench-in-the-wild HF
#   5. ScienceQA-IMG    → derek-thomas/ScienceQA HF
#   6. MMBench en+cn    → opencompass/mmbench HF (tsv with embedded images)

set -euo pipefail
ulimit -c 0

EVAL_DIR="${EVAL_DIR:-/workspace/.hf_home/eval_data}"
mkdir -p "${EVAL_DIR}"

disk_check() {
    local free=$(df -BG /workspace | awk 'NR==2{gsub("G","",$4); print $4}')
    if (( free < 30 )); then
        echo "[$(date)] ABORT: disk free ${free}G < 30G threshold" >&2
        exit 1
    fi
    echo "[$(date)] disk OK: ${free}G free"
}

PY=/usr/bin/python3

# ---------- 1. TextVQA val ----------
echo "[$(date)] === TextVQA val ==="
disk_check
TXT="${EVAL_DIR}/textvqa_val"
mkdir -p "${TXT}"
if [[ ! -f "${TXT}/TextVQA_0.5.1_val.json" ]]; then
    curl -fsSL -o "${TXT}/TextVQA_0.5.1_val.json" \
        https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
    echo "  ✓ val questions JSON"
else
    echo "  ✓ already exists"
fi
# TextVQA uses train_images for val too (already on disk under LLaVA-Instruct/images/textvqa)

# ---------- 2. POPE ----------
echo "[$(date)] === POPE ==="
disk_check
POPE="${EVAL_DIR}/pope"
mkdir -p "${POPE}"
for tag in adversarial popular random; do
    f="${POPE}/coco_pope_${tag}.json"
    if [[ ! -f "$f" ]]; then
        curl -fsSL -o "$f" \
          "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_${tag}.json"
        echo "  ✓ ${tag}"
    fi
done
# POPE uses COCO val2014. Download if missing.
if [[ ! -d "${POPE}/val2014" ]]; then
    echo "  downloading COCO val2014 (~6GB) ..."
    curl -fsSL -o "${POPE}/val2014.zip" \
      http://images.cocodataset.org/zips/val2014.zip
    ( cd "${POPE}" && unzip -q val2014.zip && rm val2014.zip )
    echo "  ✓ COCO val2014"
fi

# ---------- 3. MM-Vet ----------
echo "[$(date)] === MM-Vet ==="
disk_check
MV="${EVAL_DIR}/mm-vet"
mkdir -p "${MV}"
if [[ ! -f "${MV}/mm-vet.zip" && ! -d "${MV}/images" ]]; then
    ${PY} -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='whyu/mm-vet', repo_type='dataset',
                  local_dir='${MV}', local_dir_use_symlinks=False)
" 2>&1 | tail -5
fi
echo "  ✓ mm-vet"

# ---------- 4. LLaVA-Bench-Wild ----------
echo "[$(date)] === LLaVA-Bench-Wild ==="
disk_check
LB="${EVAL_DIR}/llava-bench-wild"
mkdir -p "${LB}"
${PY} -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='liuhaotian/llava-bench-in-the-wild', repo_type='dataset',
                  local_dir='${LB}', local_dir_use_symlinks=False)
" 2>&1 | tail -3
echo "  ✓ llava-bench-wild"

# ---------- 5. ScienceQA test (image questions only) ----------
echo "[$(date)] === ScienceQA-IMG test ==="
disk_check
SQ="${EVAL_DIR}/scienceqa"
mkdir -p "${SQ}"
${PY} -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='derek-thomas/ScienceQA', repo_type='dataset',
                  local_dir='${SQ}', local_dir_use_symlinks=False,
                  allow_patterns=['data/test*','*.json','*.jsonl'])
" 2>&1 | tail -3
echo "  ✓ scienceqa"

# ---------- 6. MMBench en + cn (tsv) ----------
echo "[$(date)] === MMBench ==="
disk_check
MB="${EVAL_DIR}/mmbench"
mkdir -p "${MB}"
for f in mmbench_dev_20230712.tsv mmbench_dev_cn_20231003.tsv; do
    if [[ ! -f "${MB}/${f}" ]]; then
        curl -fsSL -o "${MB}/${f}" \
          "https://opencompass.openxlab.space/utils/MMBench/${f}" || \
        echo "  WARN ${f} URL may differ; check opencompass mirror"
    fi
done
echo "  ✓ mmbench (best-effort)"

echo "[$(date)] === Priority A done ==="
df -BG /workspace | tail -1
du -sh "${EVAL_DIR}"/* 2>/dev/null
