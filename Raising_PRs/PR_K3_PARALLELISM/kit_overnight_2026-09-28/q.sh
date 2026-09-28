#!/bin/bash
# Overnight queue after T2, one GPU job at a time on 8 x 5060: T3, T4, T5. Progress in $Q/progress.txt.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
Q=$S/overnight; P=$Q/progress.txt; V=/workspace/venv_bfx9/bin
KL=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_pp_lowerbound_2026-09-26
SHIM=$S/lbplan/pr5_torch_compat_shim.patch
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
shim_on() { for t in "$@"; do git -C $t apply $SHIM || { note "shim failed on $t"; return 1; }; done; }
shim_off() { for t in "$@"; do git -C $t checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; echo "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)" >> $Q/worktrees_after.txt; done; }
pp() {  # <root> <name> <tree> <cache> <steps> [VAR=val ...]
  local root=$1 name=$2 tree=$3 cache=$4 n=$5; shift 5
  env "$@" LB_ROOT=$root bash $KL/run_lb.sh $name $tree $cache $n
  note "$name $(grep -o 'rc=[0-9]*' $root/$name/train.log | tail -1)"
}
pyt() {  # <name> <tree> <files...>
  local name=$1 tree=$2; shift 2
  ( cd $tree && timeout 1800 $V/python -m pytest "$@" -q -p no:cacheprovider > $Q/$name.pytest.log 2>&1 ); note "$name pytest rc=$? $(tail -1 $Q/$name.pytest.log)"
}

# ---- T3: #4656 + PR A (c87a5e102) against the dev branch (1777ad806), s6 layout, 10 steps, one warm cache
R3=$Q/t3; mkdir -p $R3; A=$S/wt_pra4656; D=$S/wt_dev
shim_on $A $D && {
  export PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_LPS=3
  C=$R3/cache0; mkdir -p $C
  pp $R3 t3_warm_pra $A $C 1 PPMEM_DIM=2048 PPMEM_SEQ=2048
  pp $R3 t3_warm_dev $D $C 1 PPMEM_DIM=2048 PPMEM_SEQ=2048
  for x in "pra:$A" "dev:$D"; do n=${x%%:*}; t=${x#*:}; rm -rf $R3/cache_$n; cp -r $C $R3/cache_$n; pp $R3 t3_mem_$n $t $R3/cache_$n 10 PPMEM_DIM=2048 PPMEM_SEQ=2048; done
  rm -rf $R3/cache*
}
shim_off $A $D
note "T3 done"

# ---- T4: #4765 (56f4cd3b0) and #4764 (0a9034257) restacked on the dev branch
O=$S/wt_o4765n; B=$S/wt_b4764n
pyt t4_4765 $O tests/unit_tests/cpu/test_activation_storage.py tests/unit_tests/cpu/test_kimi_k3_pp_memory.py tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py
pyt t4_4764 $B tests/unit_tests/cpu/test_activation_storage.py tests/unit_tests/cpu/test_kimi_k3_pp_memory.py tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py tests/unit_tests/cpu/test_kimi_k3_pp_memory_plan.py tests/unit_tests/cpu/test_activation_storage_pool.py
R4=$Q/t4; mkdir -p $R4
shim_on $D $O $B && {
  export PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_LPS=3
  C=$R4/cache_c; mkdir -p $C
  for x in "c_dev:$D:" "c_o4765all:$O:PPMEM_CPU_OFFLOAD=all" "c_b4764plan:$B:PPMEM_CPU_OFFLOAD=planned"; do
    IFS=: read -r n t e <<< "$x"; pp $R4 w$n $t $C 1 PPMEM_DIM=2048 PPMEM_SEQ=2048 $e; done
  for x in "c_dev:$D:" "c_o4765all:$O:PPMEM_CPU_OFFLOAD=all" "c_b4764plan:$B:PPMEM_CPU_OFFLOAD=planned"; do
    IFS=: read -r n t e <<< "$x"; rm -rf $R4/cache_$n; cp -r $C $R4/cache_$n; pp $R4 $n $t $R4/cache_$n 6 PPMEM_DIM=2048 PPMEM_SEQ=2048 $e; done
  C=$R4/cache_d; mkdir -p $C
  for x in "d_off:" "d_bal:PPMEM_BALANCE=1" "d_planbal:PPMEM_CPU_OFFLOAD=planned,PPMEM_BALANCE=1"; do
    IFS=: read -r n e <<< "$x"; pp $R4 w$n $B $C 1 PPMEM_DIM=1024 PPMEM_SEQ=512 PPMEM_AC=none ${e//,/ }; done
  for x in "d_off:" "d_bal:PPMEM_BALANCE=1" "d_planbal:PPMEM_CPU_OFFLOAD=planned,PPMEM_BALANCE=1"; do
    IFS=: read -r n e <<< "$x"; rm -rf $R4/cache_$n; cp -r $C $R4/cache_$n; pp $R4 $n $B $R4/cache_$n 6 PPMEM_DIM=1024 PPMEM_SEQ=512 PPMEM_AC=none ${e//,/ }; done
  rm -rf $R4/cache*
}
shim_off $D $O $B
note "T4 done"

# ---- T5: DEP bb3e38d4a
bash $Q/t5.sh
note "T5 done"
echo done > $Q/Q_DONE
