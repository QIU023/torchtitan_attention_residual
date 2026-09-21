#!/bin/bash
# tianyu's configuration as stated, in fp32: gradient accumulation (the stock trainer, four micro-batches) against PP
# with the cache off and fsdp_reshard_after_forward=always, 100 steps on the fp32 lineage cache; then the same PP cell
# with the norm reduced in the reference's order (PP_NORM_EXACT). Compared with f32_dp1 (stock GA) afterwards.
set -eu
TT=/workspace/tt_pp; OUT=/workspace/results/pp_r3; LOG=/workspace/results/pp_r3_run.log; K=/workspace/kit
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT NCCL_NVLS_ENABLE=0
cd $TT && python3 $K/probe_fp32_locate.py $TT >> $LOG 2>&1 && python3 $K/probe_pp_norm_exact.py $TT >> $LOG 2>&1 && python3 -m py_compile torchtitan/training_engine.py torchtitan/trainer.py
export GN_FP32=1 FP32_PROBE=1
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 100"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1"
F32="--training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"; P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
RA="--parallelism.fsdp_reshard_after_forward always"
cell() {  # cell <name> <gpus> <nproc> <port> <config> <flags...>
  local nm=$1 gpus=$2 np=$3 port=$4 cfg=$5; shift 5
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $OUT/seed_c4_1024/checkpoint $d/
  cp -r $OUT/ind_f32_dp1_ns_seqsum $OUT/ind_$nm; cp -r $OUT/tri_f32_dp1 $OUT/tri_$nm
  set +e
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
      torchrun --nproc_per_node=$np --master_port=$port $COMMON --config $cfg "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; set -e; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $OUT/$nm.log) $(date -u +%T)" >> $LOG
}
echo "== RF: fp32, GA vs PP cache off with fsdp_reshard_after_forward=always $(date -u +%T)" >> $LOG
cell f32_vp2n_ra 0,1 2 47401 kimi_k3_debugmodel_c4_pp_naive $B4 $P2 $IL $F32 $RA &
cell f32_pp2_ra 2,3 2 47402 kimi_k3_debugmodel_c4 $B4 $P2 $F32 $RA &
wait
( export PP_NORM_EXACT=/workspace/results/pp_r3_locate/fqn_order.txt; cell f32_vp2n_ra_xn 0,1 2 47403 kimi_k3_debugmodel_c4_pp_naive $B4 $P2 $IL $F32 $RA )
export TABLE_STEPS="1 10 50 100"
{ echo "# fp32: stock GA (f32_dp1) as the reference"; python3 $K/tables.py $OUT f32_dp1 f32_dp1_ns_oldboth f32_vp2n f32_vp2n_ra f32_pp2_ra f32_vp2n_ra_xn; } > $OUT/tables_rf.md 2>&1
echo "RF-DONE $(date -u +%T)" >> $LOG
