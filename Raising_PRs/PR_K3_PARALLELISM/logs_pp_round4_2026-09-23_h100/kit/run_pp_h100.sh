#!/bin/bash
# PR 4312 body tables on 4 x H100, head ce67cece4 on upstream main 7349a2282, the round-3 protocol: c4 text rows,
# 100 steps, seed 42, deterministic, one seed checkpoint per batch shape, fp32 total grad norm (GN_FP32), the
# reference accumulating like the pipeline (NOSYNC_GA). Each reference runs cold to fill the compile cache, then
# on a copy of its finished cache, and that second run is the reference; every other cell copies that cache.
# Phases: A bf16 1024-token, B fp32, C dp2 2048-token. Tree = ~/tt_pp with probe_apply_h100.py applied.
set -u
TT=/home/ubuntu/tt_pp; K=/home/ubuntu/kit_pp; OUT=${OUT:-/home/ubuntu/results/pp_r4}; mkdir -p $OUT
export VIRTUAL_ENV=/home/ubuntu/venv_k3 PATH=/home/ubuntu/venv_k3/bin:$PATH PYTHONPATH=$TT
export STEPS=${STEPS:-100} NCCL_NVLS_ENABLE=0
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
F32="--training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32"
D="--parallelism.data_parallel_shard_degree"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4"
C4=kimi_k3_debugmodel_c4; C4N=kimi_k3_debugmodel_c4_pp_naive
C8=kimi_k3_debugmodel_c4_8stages; C8N=kimi_k3_debugmodel_c4_8stages_naive
C16=kimi_k3_debugmodel_c4_16stages; C16N=kimi_k3_debugmodel_c4_16stages_naive

seed() {  # seed <tag> <batch flags...>
  local tag=$1; shift
  [ -d $OUT/seed_$tag/checkpoint ] && return
  ( cd $TT && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$OUT/ind_seed_$tag TRITON_CACHE_DIR=$OUT/tri_seed_$tag torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
      $COMMON --config kimi_k3_debugmodel_c4_seed --training.steps 1 "$@" --dump-folder $OUT/seed_$tag > $OUT/seed_$tag.log 2>&1 )
  echo "seed_$tag rc=$? $(date -u +%T)"
}
cell() {  # cell <name> <gpus> <nproc> <seed tag> <cache source or -> <config> <flags...>
  local nm=$1 gpus=$2 np=$3 stag=$4 csrc=$5 cfg=$6; shift 6
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $OUT/seed_$stag/checkpoint $d/
  if [ "$csrc" != "-" ] && [ -d $OUT/ind_$csrc ]; then cp -r $OUT/ind_$csrc $OUT/ind_$nm; cp -r $OUT/tri_$csrc $OUT/tri_$nm 2>/dev/null; fi
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
      $COMMON --config $cfg --training.steps $STEPS "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $OUT/$nm.log) $(date -u +%T)"
}
ref() {  # ref <name> <gpus> <nproc> <seed tag> <config> <flags...>: cold fill, then the warm run that is the reference
  local nm=$1; shift; local gpus=$1 np=$2 stag=$3 cfg=$4; shift 4
  ( export NOSYNC_GA=1; cell ${nm}_cold $gpus $np $stag - $cfg "$@" )
  ( export NOSYNC_GA=1; cell $nm $gpus $np $stag ${nm}_cold $cfg "$@" )
}
table() { python3 $K/tables.py $OUT "$@"; }

export GN_FP32=1
echo "== seeds $(date -u +%T)"
seed c4_1024 $B4
seed c4_2048 $B8

echo "== A: bf16, 1024 tokens per step (4 x 256), fp32 norm $(date -u +%T)"
ref dp1_ns 0 1 c4_1024 $C4 $B4 $D 1
cell dp1 1 1 c4_1024 dp1_ns $C4 $B4 $D 1 &
( export MB_REVERSE=1; cell dp1_rev 2 1 c4_1024 dp1_ns $C4 $B4 $D 1 ) &
wait
cell pp2 0,1 2 c4_1024 dp1_ns $C4 $B4 $D 1 $P2 &
cell vp2c 2,3 2 c4_1024 dp1_ns $C4 $B4 $D 1 $P2 $IL &
wait
cell vp2n 0,1 2 c4_1024 dp1_ns $C4N $B4 $D 1 $P2 $IL &
cell pp2vp4c 2,3 2 c4_1024 dp1_ns $C8 $B4 $D 1 $P2 $IL &
wait
cell pp2vp4n 0,1 2 c4_1024 dp1_ns $C8N $B4 $D 1 $P2 $IL
cell pp4 0,1,2,3 4 c4_1024 dp1_ns $C4 $B4 $D 1 $P4
cell pp4vp2c 0,1,2,3 4 c4_1024 dp1_ns $C4 $B4 $D 1 $P4 $IL
cell pp4vp2n 0,1,2,3 4 c4_1024 dp1_ns $C4N $B4 $D 1 $P4 $IL
cell pp4vp4c 0,1,2,3 4 c4_1024 dp1_ns $C16 $B4 $D 1 $P4 $IL
cell pp4vp4n 0,1,2,3 4 c4_1024 dp1_ns $C16N $B4 $D 1 $P4 $IL
echo A-DONE $(date -u +%T)

echo "== B: fp32 parameters, compute and reduction (FP32_PROBE) $(date -u +%T)"
export FP32_PROBE=1
ref f32_dp1_ns 0 1 c4_1024 $C4 $B4 $D 1 $F32
cell f32_dp1 1 1 c4_1024 f32_dp1_ns $C4 $B4 $D 1 $F32 &
cell f32_vp2c 2,3 2 c4_1024 f32_dp1_ns $C4 $B4 $D 1 $P2 $IL $F32 &
wait
cell f32_vp2n 0,1 2 c4_1024 f32_dp1_ns $C4N $B4 $D 1 $P2 $IL $F32
cell f32_pp4vp4c 0,1,2,3 4 c4_1024 f32_dp1_ns $C16 $B4 $D 1 $P4 $IL $F32
cell f32_pp4vp4n 0,1,2,3 4 c4_1024 f32_dp1_ns $C16N $B4 $D 1 $P4 $IL $F32
unset FP32_PROBE
echo B-DONE $(date -u +%T)

echo "== C: bf16, 2048 tokens per step (4 x 256 per rank), dp2 $(date -u +%T)"
ref d2_dp2_ns 0,1 2 c4_2048 $C4 $B8 $D 2
cell d2_pp2 0,1,2,3 4 c4_2048 d2_dp2_ns $C4 $B8 $D 2 $P2
cell d2_vp2c 0,1,2,3 4 c4_2048 d2_dp2_ns $C4 $B8 $D 2 $P2 $IL
cell d2_vp2n 0,1,2,3 4 c4_2048 d2_dp2_ns $C4N $B8 $D 2 $P2 $IL
echo C-DONE $(date -u +%T)

export TABLE_STEPS="1 10 50 100"
{
echo "# A: c4, 1024 tokens per step (4 x 256), bf16, fp32 total grad norm; reference dp1 with matched accumulation (warm)"; table dp1_ns dp1_ns_cold dp1 dp1_rev pp2 pp4 vp2n vp2c pp2vp4n pp2vp4c pp4vp2n pp4vp2c pp4vp4n pp4vp4c
echo; echo "# B: fp32 parameters, compute and reduction (FP32_PROBE), cache on and off"; table f32_dp1_ns f32_dp1_ns_cold f32_dp1 f32_vp2n f32_vp2c f32_pp4vp4n f32_pp4vp4c
echo; echo "# C: c4, 2048 tokens per step (4 x 256 per rank), dp2, bf16, fp32 norm"; table d2_dp2_ns d2_dp2_ns_cold d2_pp2 d2_vp2n d2_vp2c
} > $OUT/tables.md 2>&1
echo RUN-PP-H100-DONE $(date -u +%T)
