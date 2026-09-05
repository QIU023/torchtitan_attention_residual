#!/bin/bash
# Pure upstream/main 390e2985b (4446 merged): the multimodal debug flavor under the default backend, one step.
source /venv/main/bin/activate; cd /tmp/wt_main4446
for be in spmd_types partial_dtensor; do
  d=/workspace/main4446_check/run_$be; rm -rf $d; mkdir -p $d; cp -r --reflink=auto /workspace/.mx3_seeds_main/kimi_k3_debugmodel_mm_bf6a25965956/seed_ckpt $d/checkpoint
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/tmp/wt_main4446 TORCHINDUCTOR_CACHE_DIR=/workspace/main4446_check/ind_$be timeout 1200 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 1 --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 1 --parallelism.spmd_backend $be --dump-folder $d > /workspace/main4446_check/$be.log 2>&1
  printf "%-16s rc=%s %s\n" $be $? "$(sed 's/\x1b\[[0-9;]*m//g' /workspace/main4446_check/$be.log | grep -oE 'step: +1 +loss: +[0-9.]+' | head -1)"; sed 's/\x1b\[[0-9;]*m//g' /workspace/main4446_check/$be.log | grep -m1 -E "ValueError|Error:" | cut -c1-200
  rm -rf $d/checkpoint
done
echo "MAIN4446 CHECK DONE"
