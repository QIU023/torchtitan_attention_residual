#!/bin/bash
# MoonEP after the 09-30 smoke, where the recipe's SelectiveAC failed on real MoonEP in the recompute: the CPU tests again
# with transformers installed, then with activation checkpointing off (MOONEP_AC=none) the smoke's numerics and timing
# cells and the load cells, and the MoonEP cell once with FullAC. One GPU job at a time; progress in rerun.txt.
M=~/mep; V=$M/venv_src; T=$M/w/moonep; KS=~/kit/kit_moonep_rewrite_2026-09-29; KL=~/kit/overnight/moonep
O=$M/results/moonep_rerun; mkdir -p $O; P=$O/rerun.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "tree $(git -C $T rev-parse --short HEAD)"
. $V/bin/activate
uv pip install -q transformers > $O/pip.log 2>&1
note "transformers $(python -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1); torch $(python -c 'import torch; print(torch.__version__)')"
( cd $T && timeout 1800 python -m pytest tests/unit_tests/cpu/test_moe.py tests/unit_tests/cpu/test_integration_test_definitions.py -q \
    > $O/cpu.log 2>&1 )
note "cpu rc=$? $(tail -1 $O/cpu.log)"
cell() {  # <name> <config> <cache> <steps> [extra args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $T && PYTHONPATH=$KS:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc timeout 1800 \
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module moonep_local --config $cfg --training.steps $steps \
    --metrics.log_freq 1 --debug.seed 42 --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$name rc=$rc"
}
export MOONEP_AC=none
C0=$O/cache0; rm -rf $C0; mkdir -p $C0
for b in standard moonep standard_ep2; do cell warm_$b ${b}_cell $C0 1 --debug.deterministic; done
for x in standard:standard standard_b:standard standard_ep2:standard_ep2 moonep:moonep; do
  n=${x%%:*}; b=${x#*:}; rm -rf $O/cache_c; cp -r $C0 $O/cache_c
  cell num_$n ${b}_cell $O/cache_c 20 --debug.deterministic
done
for b in standard moonep; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; cell time_$b ${b}_cell $O/cache_c 30; done
rm -rf $O/cache_c
export MOONEP_AC=full; cell full_moonep moonep_cell $C0 20 --debug.deterministic; export MOONEP_AC=none
rm -rf $C0
python $KS/tables.py $O > $O/tables.txt 2>&1
note "smoke cells done"
MODE=real TREE=$T VENV=$V OUT=$O/load bash $KL/run_load.sh > $O/load.out 2>&1
note "load rc=$?"
echo done > $O/DONE
