#!/bin/bash
# Third attempt: bal_devp keeps the compute on the default stream and puts each destination pool in a
# pool_holder.py process on the destination GPU (K3_TEST_TCP_POOL_PROCESS=1), whose own legacy stream takes
# mooncake's copies. The second attempt's side stream crashed both side-stream cells (device-side assert).
# Second attempt (the first hung: mooncake tcp copies into a device pool run on the legacy default stream and
# queue behind the destination's compute, which waits in an NCCL barrier for the sources). Device-pool cells
# compute on a side stream (probe_lb8.py, K3_TEST_SIDE_STREAM=1); off_side shows what the side stream alone does.
# 5060 test only: #4764 224bdbaf4 balance on 8 x 5060 with the destination pools in device memory over
# mooncake tcp (tcp_device_pool_local.patch, K3_TEST_TCP_DEVICE_POOL=1), against balance off and the PR's
# host pools. Layout of the 09-28 D group: 93 layers in blocks of 12, 3 per stage, pp8 x vp4 Interleaved1F1B,
# 16 micro-batches of 512 tokens, dim 1024, no AC, seed 42, 6 steps; warm 1-step runs per config on one
# cache, each measured cell on its own copy of it. nvidia-smi sampled during each measured cell.
set -u
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=$(cd "$(dirname "$0")" && pwd); V=/workspace/venv_0928/bin
T=$S/wt_bal8; R=$S/bal8; P=$R/progress.txt; mkdir -p $R
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
export PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_LPS=3 PPMEM_DIM=1024 PPMEM_SEQ=512 PPMEM_AC=none PPMEM_STORE_TRACK=0
cell() {  # <name> <cache> <steps> [VAR=val ...]
  local name=$1 cache=$2 steps=$3; shift 3; local D=$R/$name; rm -rf $D; mkdir -p $D
  local smi=""
  if [ "$steps" -gt 1 ]; then
    ( while :; do date +%s.%N | cut -c1-14 >> $D/smi_gpu.txt; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> $D/smi_gpu.txt;
        nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits >> $D/smi_apps.txt; sleep 1; done ) &
    smi=$!
  fi
  ( cd $T && env "$@" PPMEM_OUT=$D/mem PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 900 $V/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0,7 --role rank --tee 3 --log-dir $D/logs -m torchtitan.train --module probe_lb8 --config lb_probe \
    --training.steps $steps --debug.seed 42 --debug.deterministic --metrics.log-freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 512 \
    --parallelism.pipeline-parallel-degree 8 --parallelism.pipeline-parallel-schedule Interleaved1F1B \
    --parallelism.num-pp-microbatches 16 --dump-folder $D/dump > $D/train.log 2>&1 )
  local rc=$?; [ -n "$smi" ] && kill $smi 2>/dev/null; wait $smi 2>/dev/null
  echo "rc=$rc" >> $D/train.log; rm -rf $D/dump
  note "$name rc=$rc $(grep -rh -E 'K3_TEST_TCP_POOL_PROCESS|K3_TEST_TCP_DEVICE_POOL' $D/logs 2>/dev/null | sed 's/.*rank/rank/' | sort -u | paste -sd' ') ; $(sed 's/\x1b\[[0-9;]*m//g' $D/train.log | grep -a -o 'step: *[0-9]* *loss: *[0-9.]*' | tail -1)"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
C=$R/cache_w   # warmed by the first attempt's w_off and w_bal_host (the kernels do not depend on the stream)
for x in "bal_devp:PPMEM_BALANCE=1,PPMEM_REMOTE_PROTOCOL=tcp,K3_TEST_TCP_POOL_PROCESS=1,K3_TEST_POOL_HOLDER=$K/pool_holder.py"; do
  IFS=: read -r n e <<< "$x"; rm -rf $R/cache_$n; cp -r $C $R/cache_$n
  cell $n $R/cache_$n 6 ${e//,/ }; rm -rf $R/cache_$n
done
note "bal8c done"
