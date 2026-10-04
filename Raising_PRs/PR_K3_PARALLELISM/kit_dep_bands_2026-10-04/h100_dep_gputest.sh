#!/bin/bash
# The DEP NCCL unit test on 8518f473f (the PR head), 4 x H100, after the MoonEP round finishes.
M=~/mep; V=$M/venv_src; O=$M/results/dep_fig11_gputest; mkdir -p $O
until [ -f $M/results/moonep_perf/DONE ]; do sleep 30; done
( cd $M/w/dep_fig11 && echo "$(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)" > $O/tree.txt && \
  timeout 1800 $V/bin/python -m pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q -rA > $O/pytest.log 2>&1; echo "rc=$?" > $O/rc )
echo done > $O/DONE
