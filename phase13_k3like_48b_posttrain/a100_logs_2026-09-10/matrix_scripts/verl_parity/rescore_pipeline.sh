#!/bin/bash
# Wait for verl's log-prob batch dump, stop the cell, rescore offline with torchtitan (GPU 0) and vLLM (GPU 1), compare.
D=/workspace/logprob_dump2; S=$(dirname "$0")
until [ -f $D/batch.pt ] || grep -q "^rc=" /workspace/verl_newtree_lpdump2_run.log 2>/dev/null; do sleep 20; done; sleep 10
P2=$(ps -eo pid,ppid,cmd | awk '$2==1 && /bash .*verl_grpo_newtree_reward.sh/ {print $1}')
for r in $P2; do if grep -q "lpdump2" /proc/$r/environ 2>/dev/null; then kids() { for c in $(ps -o pid= --ppid "$1"); do kids "$c"; echo "$c"; done; }; T=$(kids $r); kill -9 $T $r 2>/dev/null; echo "killed runner $r"; fi; done
sleep 3
for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -f <(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | sed -n 7,8p | cut -d, -f2 | tr -d ' ') | cut -d, -f1); do kill -9 $p 2>/dev/null; done
ls -la $D/batch.pt
source /workspace/venv_verl/bin/activate; export PYTHONPATH=/tmp/wt_k3int_cp:/tmp/attn_gym_up HF_HOME=/workspace/.hf_home FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_rescore TRITON_CACHE_DIR=/workspace/.triton_rescore
CUDA_VISIBLE_DEVICES=0 python $S/rescore_tt.py 2>&1 | grep -v "W09\|WARNING\|deprecat" | tail -8
CUDA_VISIBLE_DEVICES=1 EAGER=1 PC=1 TAG=_eager_pc python $S/rescore_vllm.py 2>&1 | grep -v "W09\|WARNING\|deprecat\|INFO" | tail -3
CUDA_VISIBLE_DEVICES=1 EAGER=1 PC=0 TAG=_eager_nopc python $S/rescore_vllm.py 2>&1 | grep -v "W09\|WARNING\|deprecat\|INFO" | tail -3
for t in _eager_pc _eager_nopc; do echo "=== vLLM offline variant $t"; TAG=$t python $S/rescore_compare.py 2>&1 | grep -v "W09\|WARNING\|deprecat"; done
echo RESCORE-DONE
