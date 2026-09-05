#!/bin/bash
# The perturbation control: the same float32 grad-norm change that moves K3's dp1 by 3.2 percent at
# step 10 (3.45908 -> 3.34752), applied to upstream's flavors on pure main (worktree /tmp/wt_main_gn:
# 390e2985b + local_hacks/gn_fp32_norm.patch). Same seeds and protocol as ctl_pp_matrix.sh; GPU 7.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_ctl CUDA_VISIBLE_DEVICES=7
D="--parallelism.data_parallel_shard_degree"
PD="--parallelism.spmd_backend partial_dtensor --training.disable_cuda_graphs"
export SEED_EXTRA="$PD"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
for m in llama3 deepseek_v3; do
MODULE=$m SEED_CFG=${m}_debugmodel TITAN=/tmp/wt_main_gn CFG=${m}_debugmodel BATCH="$B" \
CELLS="dp1_gn|1|$D 1 $PD" $MX ctlgn_$m
done
echo "CTL GN MATRIX DONE"
