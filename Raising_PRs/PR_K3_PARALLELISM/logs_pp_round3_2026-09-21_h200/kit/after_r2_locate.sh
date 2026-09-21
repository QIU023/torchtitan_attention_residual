#!/bin/bash
# After the warm self-reruns (RERUN-ALL-DONE), apply the fp32 locate hacks to the tree and run the locate campaign at 100 steps.
set -u
LOG=/workspace/results/pp_r3_run.log
until grep -q 'RERUN-ALL-DONE' $LOG; do sleep 5; done
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH
python3 /workspace/kit/probe_fp32_locate.py /workspace/tt_pp >> $LOG 2>&1 || { echo "LOCATE-PATCH-FAILED" >> $LOG; exit 1; }
python3 -m py_compile /workspace/tt_pp/torchtitan/training_engine.py /workspace/tt_pp/torchtitan/trainer.py || { echo "LOCATE-PATCH-SYNTAX-Error" >> $LOG; exit 1; }
echo "== L: fp32 locate campaign, 100 steps $(date -u +%T)" >> $LOG
STEPS=100 bash /workspace/kit/run_fp32_locate.sh >> $LOG 2>&1
