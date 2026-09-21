#!/bin/bash
# tianyu's configuration: PP cache off with fsdp_reshard_after_forward=always (the reference, no PP, already reshards by
# default), bf16, fp32 total norm, no NOSYNC_GA on either side, 100 steps on the first lineage cache. Compared later against
# dp1 (stock accumulation) and dp1_ns_oldboth (matched accumulation).
set -u
TT=/workspace/tt_pp; OUT=/workspace/results/pp_r3
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT NCCL_NVLS_ENABLE=0 GN_FP32=1
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 100"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"; P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
RA="--parallelism.fsdp_reshard_after_forward always"
cell() {  # cell <name> <gpus> <nproc> <port> <config> <flags...>
  local nm=$1 gpus=$2 np=$3 port=$4 cfg=$5; shift 5
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $OUT/seed_c4_1024/checkpoint $d/
  cp -r $OUT/ind_dp1_ns_seqsum $OUT/ind_$nm; cp -r $OUT/tri_dp1 $OUT/tri_$nm
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
      torchrun --nproc_per_node=$np --master_port=$port $COMMON --config $cfg "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $OUT/$nm.log) $(date -u +%T)" >> /workspace/results/pp_r3_run.log
}
echo "== RA: PP cache off with fsdp_reshard_after_forward=always $(date -u +%T)" >> /workspace/results/pp_r3_run.log
cell vp2n_ra 0,1 2 47101 kimi_k3_debugmodel_c4_pp_naive $B4 $P2 $IL $RA &
cell pp2_ra 2,3 2 47102 kimi_k3_debugmodel_c4 $B4 $P2 $RA &
wait
echo "RA-DONE $(date -u +%T)" >> /workspace/results/pp_r3_run.log
