#!/bin/bash
# After LOCATE-DONE: the compare of the locate campaign, then the exact-norm hack on the tree and the xn cells.
set -u
LOG=/workspace/results/pp_r3_run.log; OUT=/workspace/results/pp_r3_locate
until grep -q 'LOCATE-DONE' $LOG; do sleep 5; done
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH
python3 /workspace/kit/locate_compare.py $OUT ref ref2 vp2n vp2c pp4vp4n > $OUT/compare.txt 2>&1
python3 /workspace/kit/probe_pp_norm_exact.py /workspace/tt_pp >> $LOG 2>&1 || { echo "XN-PATCH-FAILED" >> $LOG; exit 1; }
python3 -m py_compile /workspace/tt_pp/torchtitan/training_engine.py || { echo "XN-PATCH-SYNTAX-Error" >> $LOG; exit 1; }
echo "== X: fp32 whole stack cells with the norm in the reference's order, 100 steps $(date -u +%T)" >> $LOG
STEPS=100 bash /workspace/kit/run_fp32_xn.sh >> $LOG 2>&1
