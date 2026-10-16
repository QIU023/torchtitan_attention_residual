#!/bin/bash
# 10-16 verification of the current heads on 8 x 5060: video PR k3_video, then the stack. Each result is compared with
# its earlier run on a copy of that run's cache lineage. No deletion.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$S/verify_1016; PR=$K/progress.txt
VT=$S/wt_k3_video; ST=$S/wt_mxqat_fix; KS=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_verify_stack_2026-10-13
note() { echo "$(date -u +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
note "video $(git -C $VT rev-parse --short HEAD) dirty=$(git -C $VT status --short | wc -l); stack $(git -C $ST rev-parse --short HEAD) dirty=$(git -C $ST status --short | wc -l)"
# V1: the PR's GPU test file
( cd $VT && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$S/main_pins_1015:$VT TORCHINDUCTOR_CACHE_DIR=$K/cache_v1/ic TRITON_CACHE_DIR=$K/cache_v1/tc \
  timeout 1800 python -m pytest -q -p no:cacheprovider tests/unit_tests/gpu/models/kimi_k3/test_kimi_k3.py > $K/v1_gpu_test.txt 2>&1; echo "rc=$?" >> $K/v1_gpu_test.txt )
note "V1 gpu test: $(grep -E 'passed|failed' $K/v1_gpu_test.txt | tail -1)"
# V2: the parity script (local shims: flex for TorchTitan, fla from git main for the reference)
for f in 0 8 5; do
  ( cd $VT && K3_FLEX=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$S/k3v_run/flexshim:$S/parity_pins_flamain:$S/parity_pins:$S/main_pins_1015:$VT \
    TORCHINDUCTOR_CACHE_DIR=$K/cache_v2/ic TRITON_CACHE_DIR=$K/cache_v2/tc timeout 1800 python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3 --num_frames $f > $K/v2_parity_${f}f.log 2>&1 )
  note "V2 parity frames=$f rc=$? $(grep -E 'KL=' $K/v2_parity_${f}f.log | tail -1)"
done
# V3: the 22-cell matrix on copies of the 10-15 caches and seed
mkdir -p $K/matrix && for c in A A2 B C D; do cp -r $S/k3v_run/matrix1/cache_$c $K/matrix/cache_$c; done && cp -r $S/k3v_run/matrix1/seed $K/matrix/seed
$S/k3v_run/run_k3v.sh $VT $K/matrix $S/k3v_run/cells_k3v.txt
note "V3 matrix: $(grep -c 'rc=0 last step=10' $K/matrix/progress.txt)/22 rc0 at step 10"
# S3: stack CI cells on copies of the 10-14 warm caches, 20 steps, then the MXFP4 checkpoint integration test
A=$S/verify_stack_1013/ci_cells; mkdir -p $K/ci
cell() {  # <cfg> <gpus> <warm cache>
  local cfg=$1 gpus=$2 C=$K/ci/cache_$1 D=$K/ci/$1; cp -r $3 $C; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $ST && PROBE_STEPS=20 MODULE=stack_probe_1012 CONFIG=$cfg PYTHONPATH=$S/main_pins:$KS:$ST:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  note "S3 $cfg rc=$? last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
}
( cell mm 0,1,2,3 $A/cache_mm ) &
( cell mm_muon 4,5 $A/cache_mm_muon; cell cp_ulysses 4,5 $A/cache_cp_ulysses ) &
( cell cp_allgather 6,7 $A/cache_cp_allgather; cell mx_qat 6 $A/cache_mxqat ) &
wait
cell pp_vpp4 0,1,2,3,4,5,6,7 $A/cache_pp_vpp4
( cd $ST && PYTHONPATH=$S/main_pins:$ST:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$K/ci/cache_mx_qat/ic TRITON_CACHE_DIR=$K/ci/cache_mx_qat/tc CUDA_VISIBLE_DEVICES=0,1,2,3 \
  timeout 3000 python -m pytest -q -p no:cacheprovider tests/integration_tests/test_kimi_k3_mxfp4_checkpoint.py > $K/ci/mxfp4_checkpoint_test.log 2>&1 )
note "S3 mxfp4 checkpoint test rc=$? $(tail -1 $K/ci/mxfp4_checkpoint_test.log)"
# S2: the stack's GPU unit suite in CI's two lanes
lane() {  # <out> <marker> <gpus> <port>
  ( cd $ST && MASTER_PORT=$4 CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=$S/main_pins:$ST:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$1.cache/ic TRITON_CACHE_DIR=$1.cache/tc \
    timeout 5400 python -m pytest -q -p no:cacheprovider -rfE --continue-on-collection-errors -m "$2" --strict-markers tests/unit_tests/gpu > $1 2>&1; echo "rc=$?" >> $1 )
}
lane $K/lane1_stack.txt "not multi_gpu" 0 29621
note "S2 lane not multi_gpu: $(grep -E 'passed|failed' $K/lane1_stack.txt | tail -1)"
lane $K/lanem_stack.txt multi_gpu 0,1,2,3,4,5,6,7 29622
note "S2 lane multi_gpu: $(grep -E 'passed|failed' $K/lanem_stack.txt | tail -1)"
note done
