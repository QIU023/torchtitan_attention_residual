#!/bin/bash
# PR 4312 round 3: the body tables on 4 x H200 on the shared 17-layer debug model. C4 text rows, 100 steps,
# the 09-13 protocol: seed 42, deterministic, one seed checkpoint per batch shape, fp32 total grad norm
# (GN_FP32), the reference accumulating like the pipeline (NOSYNC_GA). Each reference runs twice: cold to
# fill the compile cache (discarded), then on a copy of its own finished cache; every other cell copies
# that cache. Phases: A bf16 1024-token, B fp32 (the cache-on rows), C dp2 2048-token, D the bf16-norm
# appendix (GN_FP32 off). Tree = /workspace/tt_pp with probe_apply_h200.py applied.
set -u
TT=/workspace/tt_pp; K=/workspace/kit; OUT=${OUT:-/workspace/results/pp_r3}; mkdir -p $OUT
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT
export STEPS=${STEPS:-100} NCCL_NVLS_ENABLE=0
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
F32="--training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32"
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
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
  echo "seed_$tag rc=$?"
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


# rerun_cells.sh <cell names...>: rerun the named cells (same arguments as run_pp_h200.sh) with a fixed, unique port
# each (PORT=<base> for parallel invocations, GPUS=<ids> overrides the GPUs). "warm:<cell>" runs the cell on a copy of
# its own finished cache and writes it as <cell>_w (the cell ran on the reference cache and compiled its own graphs
# cold; the warm run is the one to measure, and a second warm run must reproduce it).
PORT=${PORT:-41000}
cell() {  # same as run_pp_h200.sh, but the port is PORT and PORT advances
  local nm=$1${SUFFIX:-} gpus=${GPUS:-$2} np=$3 stag=$4 csrc=${CSRC:-$5} cfg=$6; shift 6
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $OUT/seed_$stag/checkpoint $d/
  if [ "$csrc" != "-" ] && [ -d $OUT/ind_$csrc ]; then cp -r $OUT/ind_$csrc $OUT/ind_$nm; cp -r $OUT/tri_$csrc $OUT/tri_$nm 2>/dev/null; fi
  PORT=$((PORT+1))
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
      torchrun --nproc_per_node=$np --master_port=$PORT \
      $COMMON --config $cfg --training.steps $STEPS "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $OUT/$nm.log) $(date -u +%T)"
}
for nm in "$@"; do
unset SUFFIX CSRC
if [ "${nm#warm:}" != "$nm" ]; then nm=${nm#warm:}; SUFFIX=_w; CSRC=$nm; fi
case $nm in
  dp1_ns) export GN_FP32=1; unset FP32_PROBE; ref dp1_ns 0 1 c4_1024 $C4 $B4 $D 1 ;;
  dp1) export GN_FP32=1; unset FP32_PROBE; cell dp1 1 1 c4_1024 dp1_ns $C4 $B4 $D 1 ;;
  dp1_rev) export GN_FP32=1; unset FP32_PROBE; ( export MB_REVERSE=1; cell dp1_rev 2 1 c4_1024 dp1_ns $C4 $B4 $D 1 ) ;;
  pp2) export GN_FP32=1; unset FP32_PROBE; cell pp2 0,1 2 c4_1024 dp1_ns $C4 $B4 $D 1 $P2 ;;
  vp2c) export GN_FP32=1; unset FP32_PROBE; cell vp2c 2,3 2 c4_1024 dp1_ns $C4 $B4 $D 1 $P2 $IL ;;
  vp2n) export GN_FP32=1; unset FP32_PROBE; cell vp2n 0,1 2 c4_1024 dp1_ns $C4N $B4 $D 1 $P2 $IL ;;
  pp2vp4c) export GN_FP32=1; unset FP32_PROBE; cell pp2vp4c 2,3 2 c4_1024 dp1_ns $C8 $B4 $D 1 $P2 $IL ;;
  pp2vp4n) export GN_FP32=1; unset FP32_PROBE; cell pp2vp4n 0,1 2 c4_1024 dp1_ns $C8N $B4 $D 1 $P2 $IL ;;
  pp4) export GN_FP32=1; unset FP32_PROBE; cell pp4 0,1,2,3 4 c4_1024 dp1_ns $C4 $B4 $D 1 $P4 ;;
  pp4vp2c) export GN_FP32=1; unset FP32_PROBE; cell pp4vp2c 0,1,2,3 4 c4_1024 dp1_ns $C4 $B4 $D 1 $P4 $IL ;;
  pp4vp2n) export GN_FP32=1; unset FP32_PROBE; cell pp4vp2n 0,1,2,3 4 c4_1024 dp1_ns $C4N $B4 $D 1 $P4 $IL ;;
  pp4vp4c) export GN_FP32=1; unset FP32_PROBE; cell pp4vp4c 0,1,2,3 4 c4_1024 dp1_ns $C16 $B4 $D 1 $P4 $IL ;;
  pp4vp4n) export GN_FP32=1; unset FP32_PROBE; cell pp4vp4n 0,1,2,3 4 c4_1024 dp1_ns $C16N $B4 $D 1 $P4 $IL ;;
  f32_dp1_ns) export GN_FP32=1 FP32_PROBE=1; ref f32_dp1_ns 0 1 c4_1024 $C4 $B4 $D 1 $F32 ;;
  f32_dp1) export GN_FP32=1 FP32_PROBE=1; cell f32_dp1 1 1 c4_1024 f32_dp1_ns $C4 $B4 $D 1 $F32 ;;
  f32_vp2c) export GN_FP32=1 FP32_PROBE=1; cell f32_vp2c 2,3 2 c4_1024 f32_dp1_ns $C4 $B4 $D 1 $P2 $IL $F32 ;;
  f32_vp2n) export GN_FP32=1 FP32_PROBE=1; cell f32_vp2n 0,1 2 c4_1024 f32_dp1_ns $C4N $B4 $D 1 $P2 $IL $F32 ;;
  f32_pp4vp4c) export GN_FP32=1 FP32_PROBE=1; cell f32_pp4vp4c 0,1,2,3 4 c4_1024 f32_dp1_ns $C16 $B4 $D 1 $P4 $IL $F32 ;;
  f32_pp4vp4n) export GN_FP32=1 FP32_PROBE=1; cell f32_pp4vp4n 0,1,2,3 4 c4_1024 f32_dp1_ns $C16N $B4 $D 1 $P4 $IL $F32 ;;
  d2_dp2_ns) export GN_FP32=1; unset FP32_PROBE; ref d2_dp2_ns 0,1 2 c4_2048 $C4 $B8 $D 2 ;;
  d2_dp2) export GN_FP32=1; unset FP32_PROBE; cell d2_dp2 0,1 2 c4_2048 d2_dp2_ns $C4 $B8 $D 2 ;;
  d2_ep2) export GN_FP32=1; unset FP32_PROBE; cell d2_ep2 2,3 2 c4_2048 d2_dp2_ns $C4 $B8 $D 2 $E 2 ;;
  d2_pp2) export GN_FP32=1; unset FP32_PROBE; cell d2_pp2 0,1,2,3 4 c4_2048 d2_dp2_ns $C4 $B8 $D 2 $P2 ;;
  d2_vp2c) export GN_FP32=1; unset FP32_PROBE; cell d2_vp2c 0,1,2,3 4 c4_2048 d2_dp2_ns $C4 $B8 $D 2 $P2 $IL ;;
  d2_vp2n) export GN_FP32=1; unset FP32_PROBE; cell d2_vp2n 0,1,2,3 4 c4_2048 d2_dp2_ns $C4N $B8 $D 2 $P2 $IL ;;
  bf_dp1_ns) unset GN_FP32 FP32_PROBE; ref bf_dp1_ns 0 1 c4_1024 $C4 $B4 $D 1 ;;
  bf_pp2) unset GN_FP32 FP32_PROBE; cell bf_pp2 0,1 2 c4_1024 bf_dp1_ns $C4 $B4 $D 1 $P2 ;;
  bf_vp2c) unset GN_FP32 FP32_PROBE; cell bf_vp2c 2,3 2 c4_1024 bf_dp1_ns $C4 $B4 $D 1 $P2 $IL ;;
  bf_vp2n) unset GN_FP32 FP32_PROBE; cell bf_vp2n 0,1 2 c4_1024 bf_dp1_ns $C4N $B4 $D 1 $P2 $IL ;;
  bf_pp4) unset GN_FP32 FP32_PROBE; cell bf_pp4 0,1,2,3 4 c4_1024 bf_dp1_ns $C4 $B4 $D 1 $P4 ;;
  bf_pp4vp4c) unset GN_FP32 FP32_PROBE; cell bf_pp4vp4c 0,1,2,3 4 c4_1024 bf_dp1_ns $C16 $B4 $D 1 $P4 $IL ;;
  bf_pp4vp4n) unset GN_FP32 FP32_PROBE; cell bf_pp4vp4n 0,1,2,3 4 c4_1024 bf_dp1_ns $C16N $B4 $D 1 $P4 $IL ;;
  bf_d2_dp2_ns) unset GN_FP32 FP32_PROBE; ref bf_d2_dp2_ns 0,1 2 c4_2048 $C4 $B8 $D 2 ;;
  bf_d2_pp2) unset GN_FP32 FP32_PROBE; cell bf_d2_pp2 0,1,2,3 4 c4_2048 bf_d2_dp2_ns $C4 $B8 $D 2 $P2 ;;
  bf_d2_vp2c) unset GN_FP32 FP32_PROBE; cell bf_d2_vp2c 0,1,2,3 4 c4_2048 bf_d2_dp2_ns $C4 $B8 $D 2 $P2 $IL ;;
  bf_d2_vp2n) unset GN_FP32 FP32_PROBE; cell bf_d2_vp2n 0,1,2,3 4 c4_2048 bf_d2_dp2_ns $C4N $B8 $D 2 $P2 $IL ;;
  *) echo "unknown cell $nm" ;;
esac
done
echo RERUN-DONE
