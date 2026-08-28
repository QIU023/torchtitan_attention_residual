#!/bin/bash
# Compile path (experiments/graph_trainer, aot_fx_trace): standard vs minimal_async_ep grads, before (upstream main) and after (fix).
S=<scratch>
source /venv/main/bin/activate; export NCCL_NVLS_ENABLE=0
COMMON="--debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 --training.disable_cuda_graphs --compile.memory_policy full"
for pair in "before:/workspace/tt_upstream_main" "after:/workspace/tt_maep_fix"; do
  tag=${pair%%:*}; T=${pair#*:}; cd $T
  for cfg in graph_trainer_deepseek_v3_debugmodel graph_trainer_deepseek_v3_debugmodel_minimal_async_ep; do
    D=/workspace/gpg_${tag}_$cfg; rm -rf $D; mkdir -p $D
    echo "### GRAPH $tag $cfg"
    GRAD_PROBE_OUT=$S/grads_graph_${tag}_$cfg.json PYTHONPATH=$T timeout 1800 torchrun --nproc_per_node=2 --master_port=41900 $S/grad_probe.py \
      --module graph_trainer.deepseek_v3 --config $cfg $COMMON --dump-folder $D > $S/gpg_${tag}_$cfg.log 2>&1
    echo "rc=$?"; grep -h "GRAD_PROBE_OK" $S/gpg_${tag}_$cfg.log | head -1 | cut -c1-170; grep -h -m2 "Error\|error:" $S/gpg_${tag}_$cfg.log | cut -c1-220
    rm -rf $D
  done
  python $S/grad_diff.py $S/grads_graph_${tag}_graph_trainer_deepseek_v3_debugmodel.json $S/grads_graph_${tag}_graph_trainer_deepseek_v3_debugmodel_minimal_async_ep.json 2>&1 | head -14
done
echo "### GRAPH_DONE"
