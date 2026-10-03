#!/bin/bash
# PR 4881 (zero-init) on db050eb3f + the change (4591fcf44, flavors.py identical to b4401a9f0), on the 8 x 5060 with
# venv_0928: the Kimi K3 GPU unit test, then the three B200 Kimi K3 cells for their recipes' 10 steps, one at a time,
# each on its own fresh compile cache. rc and the step count per run go to progress.txt.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
W=$S/wt_4881_db05; O=$S/run_4881_gpu; P=$O/progress.txt; mkdir -p $O
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. /workspace/venv_0928/bin/activate
cd $W
note "tree $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l) torch $(python -c 'import torch; print(torch.__version__)')"
timeout 1800 python -m pytest tests/unit_tests/gpu/test_kimi_k3.py -q -rA -p no:cacheprovider > $O/gpu_test.log 2>&1
note "gpu test rc=$? $(tail -1 $O/gpu_test.log)"
for x in kimi_k3_debugmodel_mm:4 kimi_k3_debugmodel_mm_muon:2 kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4:8; do
  cfg=${x%%:*}; n=${x#*:}; D=$O/$cfg; rm -rf $D; mkdir -p $D
  TORCHINDUCTOR_CACHE_DIR=$D/ic TRITON_CACHE_DIR=$D/tc NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) \
    MODULE=torchtitan_recipes.tests.suites.b200 CONFIG=$cfg timeout 2400 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1
  rc=$?
  steps=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)
  note "$cfg ngpu=$n rc=$rc last step=$steps"
  rm -rf $D/out $D/ic $D/tc
done
note "done"
echo done > $O/DONE
