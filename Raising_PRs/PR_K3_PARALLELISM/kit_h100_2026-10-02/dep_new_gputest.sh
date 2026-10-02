#!/bin/bash
# 10-02: the DEP GPU test (four ranks under NCCL) on the refactor b2a57dff7, after dep_new_numerics.sh.
M=~/mep; V=$M/venv_src; NEW=$M/w/dep_new; O=$M/results/dep_new_gputest; mkdir -p $O
until [ -f $M/results/dep_new_numerics/DONE ]; do sleep 20; done
echo "$(date +%H:%M:%S) tree $(git -C $NEW rev-parse --short HEAD) dirty=$(git -C $NEW status --short | wc -l)" > $O/progress.txt
( cd $NEW && timeout 1800 $V/bin/python -m pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q -rA > $O/gpu_test.log 2>&1 )
echo "$(date +%H:%M:%S) gpu test rc=$? $(tail -1 $O/gpu_test.log)" >> $O/progress.txt
echo done > $O/DONE
