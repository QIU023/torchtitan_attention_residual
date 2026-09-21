#!/bin/bash
# fp32 locate campaign (after the body matrix): the fp32 reference, pp2 x vp2 naive and cached, pp4 x vp4 naive, 20 steps
# each on a copy of the fp32 lineage cache (ind_f32_dp1_ns_seqsum + tri_f32_dp1), with every parameter gradient at
# step 1 before clipping, every parameter after the step-1 update, every micro batch loss at step 1 and the total
# grad norm of every step at full precision. Tree = /workspace/tt_pp with probe_fp32_locate.py applied.
set -u
TT=/workspace/tt_pp; OUT=/workspace/results/pp_r3_locate; SRC=/workspace/results/pp_r3; mkdir -p $OUT; cd $OUT
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT NCCL_NVLS_ENABLE=0
export GN_FP32=1 FP32_PROBE=1 GN_REPR=1 LOSS_REPR=1
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps ${STEPS:-20}"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1"
F32="--training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4 $IL"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 $IL"
cell() {  # cell <name> <gpus> <nproc> <port> <config> <flags...>
  local nm=$1 gpus=$2 np=$3 port=$4 cfg=$5; shift 5
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $SRC/seed_c4_1024/checkpoint $d/
  cp -r $SRC/ind_f32_dp1_ns_seqsum $OUT/ind_$nm; cp -r $SRC/tri_f32_dp1 $OUT/tri_$nm
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm GRAD_DUMP=$OUT/$nm PARAM_DUMP=$OUT/$nm \
      torchrun --nproc_per_node=$np --master_port=$port $COMMON --config $cfg "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) grads=$(ls $OUT/grads_$nm.rank*.pt 2>/dev/null | wc -l) params=$(ls $OUT/params_$nm.rank*.pt 2>/dev/null | wc -l) $(date -u +%T)"
}
( export NOSYNC_GA=1; cell ref 0 1 48001 kimi_k3_debugmodel_c4 $B4 $F32 ) &
cell vp2n 2,3 2 48002 kimi_k3_debugmodel_c4_pp_naive $B4 $P2 $F32 &
wait
cell vp2c 0,1 2 48003 kimi_k3_debugmodel_c4 $B4 $P2 $F32 &
( export NOSYNC_GA=1; cell ref2 2 1 48004 kimi_k3_debugmodel_c4 $B4 $F32 ) &
wait
cell pp4vp4n 0,1,2,3 4 48005 kimi_k3_debugmodel_c4_16stages_naive $B4 $P4 $F32
python3 /workspace/kit/locate_compare.py $OUT ref ref2 vp2n vp2c pp4vp4n > $OUT/compare.txt 2>&1
echo LOCATE-DONE $(date -u +%T)
