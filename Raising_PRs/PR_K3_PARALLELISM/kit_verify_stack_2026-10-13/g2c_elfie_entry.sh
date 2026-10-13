#!/bin/bash
# Rerun of the shapes that failed: full128 with the remat-aware finite hook, full256 with main's round-robin router.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; T=$S/wt_mxqat_fix; V=$S/verify_stack_1013
E=$V/elfie_entry_run6; FIX=$V/elfie_entry/fixture
mkdir -p $E; . /workspace/venv_1006/bin/activate
export PYTHONPATH=$S/main_pins:$T:$S/elfie_ao
export NCCL_RUNTIME_CONNECT=0 NCCL_NET_PLUGIN=none PYTORCH_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
note() { echo "$(date -u +%H:%M:%S) $*" >> $E/progress.txt; }
cd $T
note "tree $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l) (validate_distmuon.py ported, finite hook remat-aware)"
run() {  # <name> <gpus> <args...>
  local name=$1 n=$2; shift 2; mkdir -p $E/$name
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) timeout 2400 torchrun --nproc-per-node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    scripts/validation/kimi_k3/validate_distmuon.py --checkpoint $FIX --output $E/$name/out "$@" > $E/$name/run.log 2>&1
  local rc=$?
  note "$name rc=$rc pass_ranks=$(grep -o 'KIMI_DISTMUON_PASS rank=[0-9]*' $E/$name/run.log | sort -u | wc -l)/$n errors=$(grep -cE '(Error|Exception):' $E/$name/run.log)"
}
run full128_mini 8 --ep 4 --pp 1 --seq-len 16 --steps 10 --profile-skip-steps 5 --dispatcher standard --activation-checkpoint selective --diagnose-steps 2
F256="--seq-len 2048 --microbatch-size 2 --microbatches 8 --steps 10 --profile-skip-steps 5 --dispatcher standard --fused-loss --balanced-routing --activation-checkpoint recompute-matmuls --reshard-after-backward --diagnose-steps 2 --pipeline-startup-timeout 1800 --pipeline-steady-timeout 600"
run full256_mini_pp2_ep4 8 --ep 4 --pp 2 $F256
run full256_mini_pp8 8 --ep 1 --pp 8 $F256
note done; echo done > $E/DONE
