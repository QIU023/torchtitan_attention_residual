#!/usr/bin/env bash
# Download LLaVA-1.5 Visual Instruction Tuning data (Stage 2):
#   1. llava_v1_5_mix665k.json from liuhaotian/LLaVA-Instruct-150K (HF)
#   2. COCO train2017 images
#   3. GQA images
#   4. OCR-VQA images
#   5. TextVQA train images
#   6. Visual Genome part 1 + part 2
#
# Total ~65-70GB on disk. Layout matches the JSON's "image" field paths:
#   /workspace/.hf_home/LLaVA-Instruct/
#     llava_v1_5_mix665k.json
#     images/
#       coco/train2017/000000000009.jpg  ...
#       gqa/images/n123456.jpg  ...
#       ocr_vqa/images/<id>.jpg ...
#       textvqa/train_images/<id>.jpg ...
#       vg/VG_100K/<id>.jpg  +  VG_100K_2/<id>.jpg
#
# Resumable: every wget uses -c (continue partial), every extraction
# uses unzip -n (never overwrite). Safe to re-run.

set -uo pipefail

DEST="${DEST:-/workspace/.hf_home/LLaVA-Instruct}"
IMG="${DEST}/images"
LOG="${LOG:-${DEST}/download.log}"
mkdir -p "${IMG}/coco" "${IMG}/gqa" "${IMG}/ocr_vqa" "${IMG}/textvqa" "${IMG}/vg"
exec >>"${LOG}" 2>&1

echo ""
echo "============================================================"
echo "[$(date)] starting Instruct-665K download to ${DEST}"
echo "============================================================"

free_gb() { df -BG /workspace | awk 'NR==2{gsub("G","",$4);print $4}'; }

step() {
    local name="$1"
    echo ""
    echo "--- [$(date)] STEP: ${name} (free=$(free_gb)G) ---"
}

abort_if_disk_low() {
    local free=$(free_gb)
    if (( free < 50 )); then
        echo "[$(date)] ABORT: disk free=${free}G < 50G threshold"
        exit 1
    fi
}

# ---- 1. mix665k JSON ----
step "mix665k JSON"
if [[ ! -f "${DEST}/llava_v1_5_mix665k.json" ]]; then
    /usr/bin/python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='liuhaotian/LLaVA-Instruct-150K',
    filename='llava_v1_5_mix665k.json',
    repo_type='dataset',
    local_dir='${DEST}',
)
print('downloaded:', p)
"
    abort_if_disk_low
else
    echo "already present"
fi

# ---- 2. COCO train2017 (~18 GB) ----
step "COCO train2017"
COCO_ZIP="${IMG}/coco/train2017.zip"
if [[ ! -d "${IMG}/coco/train2017" ]]; then
    wget -c -O "${COCO_ZIP}" http://images.cocodataset.org/zips/train2017.zip
    abort_if_disk_low
    unzip -n -q "${COCO_ZIP}" -d "${IMG}/coco/"
    rm -f "${COCO_ZIP}"
    abort_if_disk_low
else
    echo "already present"
fi

# ---- 3. GQA (~20 GB) ----
step "GQA images"
GQA_ZIP="${IMG}/gqa/images.zip"
if [[ ! -d "${IMG}/gqa/images" || -z "$(ls -A "${IMG}/gqa/images" 2>/dev/null)" ]]; then
    wget -c -O "${GQA_ZIP}" https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
    abort_if_disk_low
    unzip -n -q "${GQA_ZIP}" -d "${IMG}/gqa/"
    rm -f "${GQA_ZIP}"
    abort_if_disk_low
else
    echo "already present"
fi

# ---- 4. OCR-VQA (~10 GB) ----
# Official site is broken; community mirror via HuggingFace dataset.
step "OCR-VQA images"
if [[ ! -d "${IMG}/ocr_vqa/images" || -z "$(ls -A "${IMG}/ocr_vqa/images" 2>/dev/null)" ]]; then
    /usr/bin/python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='howard-hou/OCR-VQA',
    repo_type='dataset',
    local_dir='${IMG}/ocr_vqa',
    allow_patterns=['images/*', 'images.zip'],
)
print('downloaded to:', p)
"
    # If they shipped a zip, extract it.
    if [[ -f "${IMG}/ocr_vqa/images.zip" ]]; then
        unzip -n -q "${IMG}/ocr_vqa/images.zip" -d "${IMG}/ocr_vqa/"
        rm -f "${IMG}/ocr_vqa/images.zip"
    fi
    abort_if_disk_low
else
    echo "already present"
fi

# ---- 5. TextVQA (~6 GB) ----
step "TextVQA train images"
TEXTVQA_ZIP="${IMG}/textvqa/train_val_images.zip"
if [[ ! -d "${IMG}/textvqa/train_images" || -z "$(ls -A "${IMG}/textvqa/train_images" 2>/dev/null)" ]]; then
    wget -c -O "${TEXTVQA_ZIP}" https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
    abort_if_disk_low
    unzip -n -q "${TEXTVQA_ZIP}" -d "${IMG}/textvqa/"
    rm -f "${TEXTVQA_ZIP}"
    abort_if_disk_low
else
    echo "already present"
fi

# ---- 6. Visual Genome (part 1 + part 2, ~15 GB) ----
step "Visual Genome"
VG1_ZIP="${IMG}/vg/images.zip"
VG2_ZIP="${IMG}/vg/images2.zip"
if [[ ! -d "${IMG}/vg/VG_100K" || -z "$(ls -A "${IMG}/vg/VG_100K" 2>/dev/null)" ]]; then
    wget -c -O "${VG1_ZIP}" https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
    abort_if_disk_low
    unzip -n -q "${VG1_ZIP}" -d "${IMG}/vg/"
    rm -f "${VG1_ZIP}"
    abort_if_disk_low
fi
if [[ ! -d "${IMG}/vg/VG_100K_2" || -z "$(ls -A "${IMG}/vg/VG_100K_2" 2>/dev/null)" ]]; then
    wget -c -O "${VG2_ZIP}" https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
    abort_if_disk_low
    unzip -n -q "${VG2_ZIP}" -d "${IMG}/vg/"
    rm -f "${VG2_ZIP}"
    abort_if_disk_low
else
    echo "already present"
fi

echo ""
echo "============================================================"
echo "[$(date)] ALL DONE — free=$(free_gb)G"
du -sh "${IMG}"/* 2>/dev/null
echo "============================================================"
