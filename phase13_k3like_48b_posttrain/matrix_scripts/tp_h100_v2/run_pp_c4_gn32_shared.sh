#!/bin/bash
# The fp32 grad-norm cells again, every cell of a stream on ONE inductor / triton cache, warmed by a 1-step run of
# each configuration before the 100-step runs (the per-cell cold caches of run_pp_c4_gn32.sh / run_pp_c4_d2_gn32.sh
# gave dp2 x pp2 14.4183 and 14.4192 at step 1; on a shared warm cache it reads the reference's 14.4170 every time).
# Sequential: one cache, one writer. OUT = run_pp_c4.sh's.
set -u; export CFG=kimi_k3_debugmodel_c4 GN_FP32=1; . "$(dirname "$0")/common.sh"
P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 $IL"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
cs() {  # cs <name> <steps> <seed tag> <gpus> <nproc> <common> <flags...>: one shared cache per stream ($J)
  local nm=$1 steps=$2 stag=$3 gpus=$4 np=$5 common=$6; shift 6; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r $OUT/seed_$stag/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $common --training.steps $steps "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? steps=$(grep -a -c 'step: ' $OUT/$nm.log)"; rm -rf $d/checkpoint; }
run_stream() {  # run_stream <prefix> <seed tag> <batch flags>, then the cell list as "name|gpus|nproc|common-var|env|flags" lines on stdin
  local pre=$1 stag=$2 batch=$3 line
  mapfile -t CELLS
  for steps in 1 100; do
    for line in "${CELLS[@]}"; do
      IFS='|' read -r nm gpus np cv envs flags <<< "$line"
      [ "$steps" = 1 ] && nm="warm_$nm"
      ( [ -n "$envs" ] && export $envs; cs ${pre}_$nm $steps $stag $gpus $np "${!cv}" $batch $flags )
    done
  done
}
J=$OUT/jitwarm_sh1024; mkdir -p $J
run_stream sh c4_1024 "--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256" <<LIST
dp1_ns|0|1|COMMON|NOSYNC_GA=1|$D 1
pp2|0,1|2|COMMON||$D 1 $P2
vp2n|0,1|2|NAIVE||$D 1 $P2 $IL
vp2c|0,1|2|COMMON||$D 1 $P2 $IL
pp4vp4n|0,1,2,3|4|NAIVE|PP_STAGES_PER_RANK=4|$D 1 $P4
pp4vp4c|0,1,2,3|4|COMMON|PP_STAGES_PER_RANK=4|$D 1 $P4
LIST
J=$OUT/jitwarm_gd
run_stream sh c4_2048 "--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256" <<LIST
d2_dp2_ns|0,1|2|COMMON|NOSYNC_GA=1|$D 2
d2_pp2|0,1,2,3|4|COMMON||$D 2 $P2
d2_vp2n|0,1,2,3|4|NAIVE||$D 2 $P2 $IL
d2_vp2c|0,1,2,3|4|COMMON||$D 2 $P2 $IL
LIST
export TABLE_STEPS="1 2 3 10 20 100"
echo; echo '# 1024, fp32 grad norm, one shared warm cache'; table sh_dp1_ns sh_pp2 sh_vp2n sh_vp2c sh_pp4vp4n sh_pp4vp4c
echo; echo '# 2048 dp2, fp32 grad norm, one shared warm cache'; table sh_d2_dp2_ns sh_d2_pp2 sh_d2_vp2n sh_d2_vp2c
echo RUN-GN32-SHARED-DONE
