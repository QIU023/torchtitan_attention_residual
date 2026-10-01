#!/bin/bash
# 10-01 MoonEP SAC fix on 8 x 5060 with the fake MoonEP: the h100 cell against the same cell on the standard backend
# under the recipe's SelectiveAC and under FullAC, 10 steps each on one warm cache (a step of each backend), the two
# backends at once on GPUs 0-3 and 4-7; then the MoonEP cell with SPMD type checking on (AC off), 3 steps.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_moonep_rewrite_2026-09-29
F=$K/local/fake_moonep; E=${E:-$S/moonep_sac/e2e}; W=${W:-$S/wt_moonep_0929n}; V=/workspace/venv_0928/bin
P=$E/progress.txt; mkdir -p $E
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "tree $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"
run() {  # <name> <config> <cache> <steps> <gpus> <MOONEP_AC>
  local name=$1 cfg=$2 cache=$3 steps=$4 gpus=$5 ac=$6; local D=$E/$name; rm -rf $D; mkdir -p $D
  ( cd $W && MOONEP_AC=$ac PYTHONPATH=$F:$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus timeout 1200 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d \
    --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module moonep_local --config $cfg \
    --training.steps $steps --metrics.log_freq 1 --debug.seed 42 --debug.deterministic --dump-folder $D/out \
    > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out; note "$name rc=$rc"
}
for ac in sac full; do
  a=$ac; [ "$ac" = sac ] && a=""
  C0=$E/cache_$ac; rm -rf $C0; mkdir -p $C0
  run warmstd_$ac standard_cell $C0 1 0,1,2,3 "$a"; run warmmoon_$ac moonep_cell $C0 1 0,1,2,3 "$a"
  rm -rf $C0.s $C0.m; cp -r $C0 $C0.s; cp -r $C0 $C0.m
  run std_$ac standard_cell $C0.s 10 0,1,2,3 "$a" & run moon_$ac moonep_cell $C0.m 10 4,5,6,7 "$a" & wait
  rm -rf $C0 $C0.s $C0.m
done
C1=$E/cache_tc; rm -rf $C1; mkdir -p $C1
run moon_typecheck moonep_cell $C1 3 0,1,2,3 typecheck
rm -rf $C1
note "e2e done"; echo done > $E/DONE
