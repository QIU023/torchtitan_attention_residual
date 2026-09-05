#!/bin/bash
# Upstream's B200 CI cell (torchtitan_recipes/tests/b200.py: kimi_k3_debugmodel + _use_spmd_types(typechecking=True) + dp2)
# on pure upstream/main 390e2985b (worktree /tmp/wt_main4446: only the SM120 guard-lift hack and the
# kimi_k3_debugmodel_tc registry alias, local_hacks/registry_alias_typecheck.py). 2026-09-05: rc=1 at step 1,
# ValueError from annotate_input_spmd_types: no input_sharding entry for grid_thw / pixel_values.
source /venv/main/bin/activate; D=/workspace/main4446_check; mkdir -p $D; rm -rf $D/tc_dump
CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=/tmp/wt_main4446 TORCHINDUCTOR_CACHE_DIR=$D/ind_tc TRITON_CACHE_DIR=$D/triton_tc \
timeout 840 torchrun --nproc_per_node=2 --master_port=31777 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_tc \
  --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 2 \
  --parallelism.data_parallel_shard_degree 2 --dump-folder $D/tc_dump > $D/b200cell_main.log 2>&1
echo "rc=$?"; sed 's/\x1b\[[0-9;]*m//g' $D/b200cell_main.log | grep -E "ValueError|step: +[12] " | head -2
