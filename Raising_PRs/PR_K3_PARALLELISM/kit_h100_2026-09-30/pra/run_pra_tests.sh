#!/bin/bash
# PR A's test plan on the H100 box: the five files of body v4 on PR A (345e9e00a) and on the rebased #4656 (e66a9442b).
M=~/mep; V=$M/venv_src; O=$M/results/pra_tests; mkdir -p $O
FILES="tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py tests/unit_tests/cpu/test_kimi_k3_attention_residual_recompute.py"
for x in pra:$M/w/pra pra_base:$M/w/pra_base; do
  ( cd ${x#*:} && timeout 1800 $V/bin/python -m pytest $FILES -q > $O/${x%%:*}.log 2>&1 ); rc=$?
  echo "${x%%:*} $(git -C ${x#*:} rev-parse --short HEAD) rc=$rc $(tail -1 $O/${x%%:*}.log)" >> $O/summary.txt
done
