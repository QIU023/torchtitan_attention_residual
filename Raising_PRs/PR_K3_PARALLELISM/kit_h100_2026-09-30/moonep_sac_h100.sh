#!/bin/bash
# MoonEP SAC fix (moonep_review1 after 24458aa6e) on 4 x H100 behind an NVSwitch with real MoonEP, activation
# checkpointing as the recipes set it (the h100 cell's SelectiveAC): the GPU test (5 cases, two GPUs), the h100 cell
# through the integration runner, the numerics cells (standard twice for the floor, standard at EP 2, MoonEP; 20 steps
# on one warm cache), the timing cells (30 steps), MoonEP once under FullAC, and the load cells. One GPU job at a time.
# Before running: check out the new head in $T (git -C ~/mep/w/moonep fetch origin moonep_review1 && checkout --detach FETCH_HEAD).
M=~/mep; V=$M/venv_src; T=$M/w/moonep; KS=~/kit/kit_moonep_rewrite_2026-09-29; KL=~/kit/overnight/moonep
O=$M/results/moonep_sac; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
. $V/bin/activate
( cd $T && CUDA_VISIBLE_DEVICES=0,1 timeout 1800 python -m pytest tests/unit_tests/gpu/test_moonep.py -q -rA > $O/gpu_test.log 2>&1 )
note "gpu test rc=$? $(tail -1 $O/gpu_test.log)"
( cd $T && timeout 3600 python -m tests.integration_tests.run_tests $O/ci --test_suite h100 \
    --test_name "kimi_k3_fsdp+moonep" --ngpu 4 > $O/ci.log 2>&1 )
note "h100 cell rc=$?"
cell() {  # <name> <config> <cache> <steps> [extra args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $T && PYTHONPATH=$KS:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc timeout 1800 \
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module moonep_local --config $cfg --training.steps $steps \
    --metrics.log_freq 1 --debug.seed 42 --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$name rc=$rc"
}
unset MOONEP_AC
C0=$O/cache0; rm -rf $C0; mkdir -p $C0
for b in standard moonep standard_ep2; do cell warm_$b ${b}_cell $C0 1 --debug.deterministic; done
for x in standard:standard standard_b:standard standard_ep2:standard_ep2 moonep:moonep; do
  n=${x%%:*}; b=${x#*:}; rm -rf $O/cache_c; cp -r $C0 $O/cache_c
  cell num_$n ${b}_cell $O/cache_c 20 --debug.deterministic
done
for b in standard moonep; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; cell time_$b ${b}_cell $O/cache_c 30; done
rm -rf $O/cache_c
export MOONEP_AC=full; cell full_moonep moonep_cell $C0 20 --debug.deterministic; unset MOONEP_AC
rm -rf $C0
python $KS/tables.py $O > $O/tables.txt 2>&1
note "smoke cells done"
MODE=real TREE=$T VENV=$V OUT=$O/load bash $KL/run_load.sh > $O/load.out 2>&1
note "load rc=$?"
echo done > $O/DONE
