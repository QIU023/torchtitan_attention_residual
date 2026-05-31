#!/usr/bin/env bash
# Full-N teacher eval (Qwen3-VL-30B, max_pixels=1003520) on mmbench/gqa/pope,
# reusing the student's score modules for apples-to-apples triangle.
set -uo pipefail
export HF_HOME=/home/.hf_home CUDA_VISIBLE_DEVICES=0
VPY=/home/venv/vllm/bin/python
for b in mmbench gqa pope; do
  echo "[teacher_all] === $b (full) $(date -u +%H:%M:%S) ==="
  $VPY /home/seqkd_overnight/teacher_eval.py $b 0
done
echo "TEACHER_EVAL_ALL_DONE"
