#!/bin/bash
# One 1-step standard cell on the new head with the MixedPrecisionPolicy shim.
M=~/mep; K=~/kit/kit_moonep_perf_2026-10-03; O=$M/results/moonep_recheck_1006/shim_smoke; rm -rf $O; mkdir -p $O
. $M/venv_src/bin/activate
cd $M/w/mep_16ff && PROBE_STEPS=1 PROBE_DET=1 PYTHONPATH=$K/local/mpp_shim:$K:. TORCHINDUCTOR_CACHE_DIR=$O/ic TRITON_CACHE_DIR=$O/tc \
  NGPU=4 LOG_RANK=0,1,2,3 MODULE=moonep_probe_1003 CONFIG=standard_cell timeout 900 ./run_train.sh --output-dir $O/out > $O/run.log 2>&1
echo "rc=$?" >> $O/run.log
