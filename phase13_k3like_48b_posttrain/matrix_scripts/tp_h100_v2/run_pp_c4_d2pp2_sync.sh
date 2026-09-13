#!/bin/bash
# dp2 x pp2 (1F1B), fp32 grad norm, 3 steps: twice as is, twice with torch.cuda.synchronize() before the clip.
# Is the step-1 norm (14.4183 / 14.4192 against the reference's 14.4170) a read of gradients still being reduced?
set -u; export CFG=kimi_k3_debugmodel_c4 GN_FP32=1; . "$(dirname "$0")/common.sh"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
J=$OUT/jitwarm_gd
g() { local nm=$1; shift; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r $OUT/seed_c4_2048/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=0,1,2,3 TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton \
      torchrun --nproc_per_node=4 --master_port=$((30000+RANDOM%20000)) $COMMON --training.steps 3 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? $(grep -a -E 'step: *[0-9]+ ' $OUT/$nm.log | sed -E 's/\x1b\[[0-9;]*m//g' | grep -oE 'step: *[0-9]+ +loss: *[0-9.]+ +grad_norm: *[0-9.]+' | sort -u | tr -s ' ' | tr '\n' ';')"; rm -rf $d/checkpoint; }
g sy_d2_pp2_nosync_a $B8 $D 2 $P
g sy_d2_pp2_nosync_b $B8 $D 2 $P
( export PRECLIP_SYNC=1; g sy_d2_pp2_sync_a $B8 $D 2 $P )
( export PRECLIP_SYNC=1; g sy_d2_pp2_sync_b $B8 $D 2 $P )
echo "reference gd_d2_dp2_ns step 1: $(grep -a -E 'step: *1 ' $OUT/gd_d2_dp2_ns.log | sed -E 's/\x1b\[[0-9;]*m//g' | grep -oE 'grad_norm: *[0-9.]+' | head -1)"
echo RUN-D2PP2-SYNC-DONE
