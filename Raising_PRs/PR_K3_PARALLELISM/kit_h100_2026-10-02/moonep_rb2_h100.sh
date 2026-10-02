#!/bin/bash
# 10-02: MoonEP rebased onto upstream main db050eb3f (moonep_review1 = cec583a44; the new main compiles SiTU-GLU by
# default, inside MoonEP's expert op), on the 09-30 H100 with real MoonEP 33327eb. In order, one GPU job at a time:
# the GPU test (5 cases, two GPUs), the h100 cell through the integration runner, the numerics cells (standard twice,
# standard at EP 2, MoonEP; 20 steps, deterministic, each on a copy of one warm cache), MoonEP under FullAC, and the
# timing cells (30 steps, not deterministic). Probe recipes: moonep_probe_1002.py (kit only).
M=~/mep; V=$M/venv_src; T=$M/w/moonep; K=~/kit/kit_h100_2026-10-02
O=$M/results/moonep_rb2; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
. $V/bin/activate
( cd $T && CUDA_VISIBLE_DEVICES=0,1 timeout 1800 python -m pytest tests/unit_tests/gpu/test_moonep.py -q -rA > $O/gpu_test.log 2>&1 )
note "gpu test rc=$? $(tail -1 $O/gpu_test.log)"
rm -rf $O/ci
( cd $T && timeout 3600 python -m tests.integration_tests.run_tests $O/ci --test_suite h100 \
    --test_name "kimi_k3_fsdp+moonep" --gpu_arch h100 --ngpu 4 > $O/ci.log 2>&1 )
note "h100 cell rc=$?"
cell() {  # <name> <config> <cache> <steps> <det>
  local name=$1 cfg=$2 cache=$3; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $T && PROBE_STEPS=$4 PROBE_DET=$5 PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 1800 torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module moonep_probe_1002 --config $cfg --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$name rc=$rc"
}
C0=$O/cache0; rm -rf $C0; mkdir -p $C0
for b in standard moonep standard_ep2; do cell warm_$b ${b}_cell $C0 1 1; done
for x in standard:standard standard_b:standard standard_ep2:standard_ep2 moonep:moonep moonep_full:moonep_full; do
  n=${x%%:*}; b=${x#*:}; rm -rf $O/cache_c; cp -r $C0 $O/cache_c
  cell num_$n ${b}_cell $O/cache_c 20 1
done
for b in standard moonep; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; cell time_$b ${b}_cell $O/cache_c 30 0; done
rm -rf $O/cache_c $C0
note "all cells done"
echo done > $O/DONE
