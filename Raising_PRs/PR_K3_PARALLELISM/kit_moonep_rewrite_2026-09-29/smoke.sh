#!/bin/bash
# Tonight's MoonEP smoke on 4 x H100 behind an NVSwitch (run check_box.sh and setup_box.sh first).
# 1. the PR's tests; 2. the CI cell as run_tests runs it; 3. numerics: standard twice, standard at EP 2
# (the same data under another reduction order), MoonEP
# and (if deep_ep imports) DeepEP, deterministic, one warm compile cache, 20 steps; 4. timing:
# the same backends without --debug.deterministic, 30 steps; 5. the tables.
# Env (09-30 box): VENV (default ~/mep/venv), TREE (default ~/mep/tt), MOONEP (default ~/mep/MoonEP), OUT (default ~/mep/results).
M=~/mep; K=$(cd "$(dirname "$0")" && pwd); O=${OUT:-$M/results}; mkdir -p $O
. ${VENV:-$M/venv}/bin/activate; cd ${TREE:-$M/tt}
note() { echo "$(date +%H:%M:%S) $*" | tee -a $O/summary.txt; }
note "tree $(git rev-parse --short HEAD) moonep $(git -C ${MOONEP:-$M/MoonEP} rev-parse --short HEAD)"
timeout 1800 python -m pytest tests/unit_tests/cpu/test_moe.py \
  tests/unit_tests/cpu/test_integration_test_definitions.py -q > $O/cpu.log 2>&1
note "cpu rc=$? $(tail -1 $O/cpu.log)"
timeout 900 python -m pytest tests/unit_tests/gpu/test_moonep.py -q > $O/gpu.log 2>&1
note "gpu rc=$? $(tail -1 $O/gpu.log)"
timeout 1800 python -m tests.integration_tests.run_tests $O/ci --test_suite h100 \
  --test_name "kimi_k3_fsdp+moonep" --ngpu 4 > $O/ci.log 2>&1
note "ci cell rc=$?"
cell() {  # <name> <config> <cache> <steps> [extra args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc timeout 1800 \
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module moonep_local --config $cfg --training.steps $steps \
    --metrics.log_freq 1 --debug.seed 42 --dump-folder $D/out "$@" > $D/run.log 2>&1
  note "$name rc=$?"
}
backends="standard moonep"
python -c "import deep_ep" 2>/dev/null && backends="$backends deepep"
C0=$O/cache0; rm -rf $C0; mkdir -p $C0
for b in $backends standard_ep2; do cell warm_$b ${b}_cell $C0 1 --debug.deterministic; done
for x in standard:standard standard_b:standard standard_ep2:standard_ep2 $(for b in $backends; do [ $b != standard ] && echo $b:$b; done); do
  n=${x%%:*}; b=${x#*:}; rm -rf $O/cache_c; cp -r $C0 $O/cache_c
  cell num_$n ${b}_cell $O/cache_c 20 --debug.deterministic
done
for b in $backends; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; cell time_$b ${b}_cell $O/cache_c 30; done
rm -rf $O/cache_c
python $K/tables.py $O | tee $O/tables.txt
note "smoke done"
