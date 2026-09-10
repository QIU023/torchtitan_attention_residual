#!/bin/bash
# deepseek_v3_debugmodel, plain GroupedExperts, ep2 x fsdp2, full AC, 10 steps: standard vs
# minimal_async_ep, before (upstream main) and after (maep_dispatch_owned) the fix. Same seed init.
S=<scratch>
source /venv/main/bin/activate; export NCCL_NVLS_ENABLE=0
COMMON="--debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2"
for pair in "before:/workspace/tt_upstream_main" "after:/workspace/tt_maep_fix"; do
  tag=${pair%%:*}; T=${pair#*:}; cd $T
  for cfg in dsv3_std dsv3_maep_plain; do
    D=/workspace/d10_${tag}_$cfg; rm -rf $D
    PYTHONPATH=$T:$S timeout 1500 torchrun --nproc_per_node=2 --master_port=41700 -m torchtitan.train --module dsv3_probe --config $cfg $COMMON --dump-folder $D > $S/d10_${tag}_$cfg.log 2>&1
    rc=$?; L=$S/d10_${tag}_$cfg.log
    printf "%-6s %-16s rc=%s" $tag $cfg $rc
    for st in 1 2 3 10; do v=$(grep -oE "step: *$st .*loss: *[0-9.]+" $L | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+'); g=$(grep -oE "step: *$st .*grad_norm: *[0-9.]+" $L | head -1 | grep -oE 'grad_norm: *[0-9.]+' | grep -oE '[0-9.]+'); printf "  s%s=%s(g%s)" $st "$v" "$g"; done; echo
    rm -rf $D
  done
done
echo "### D10_DONE"
