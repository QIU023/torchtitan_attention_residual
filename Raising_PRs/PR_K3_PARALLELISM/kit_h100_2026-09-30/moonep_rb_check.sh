#!/bin/bash
# 10-01: the MoonEP branch after its port onto upstream main 1aaee42bf (e30d82886), on the 09-30 H100 with real
# MoonEP: the GPU test (5 cases, two GPUs) and the h100 cell through the integration runner.
M=~/mep; V=$M/venv_src; T=$M/w/moonep; O=$M/results/moonep_rb; mkdir -p $O; P=$O/progress.txt
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
echo done > $O/DONE
