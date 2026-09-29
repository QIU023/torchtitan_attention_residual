#!/bin/bash
# CPU tests of the PP trees on the source-built torch (PR A overrides torch's private stage methods).
M=~/mep; . $M/venv_src/bin/activate; O=$M/results/cpu_tests; mkdir -p $O
PP="tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py"
declare -A T=(
  [main_5dc9]="$PP"
  [pra]="$PP"
  [dev]="$PP tests/unit_tests/cpu/test_kimi_k3_attention_residual_recompute.py"
  [o4765]="$PP tests/unit_tests/cpu/test_activation_storage.py tests/unit_tests/cpu/test_kimi_k3_pp_memory.py"
  [b4764]="$PP tests/unit_tests/cpu/test_activation_storage.py tests/unit_tests/cpu/test_activation_storage_pool.py tests/unit_tests/cpu/test_kimi_k3_pp_memory.py tests/unit_tests/cpu/test_kimi_k3_pp_memory_plan.py"
)
for t in main_5dc9 pra dev o4765 b4764; do
  ( cd $M/w/$t && CUDA_VISIBLE_DEVICES= timeout 1800 python -m pytest ${T[$t]} -q -p no:cacheprovider > $O/$t.log 2>&1 )
  echo "$(date +%H:%M:%S) $t $(git -C $M/w/$t rev-parse --short HEAD) rc=$? $(tail -1 $O/$t.log)" >> $O/summary.txt
done
echo done >> $O/summary.txt
