#!/bin/bash
# The CP matrix on cp_review5 edc4cd71b (the declarations PR + the stack copy + the CP layer on the
# merged Attention Gym recipe), Attention Gym at upstream/main b19162e (worktree /tmp/attn_gym_up,
# put first on PYTHONPATH): the eleven cells of the body plus, for every cell with a dp axis, its
# expert-parallel twin, and dp x cp at world 8.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up
S="--parallelism.spmd_backend spmd_types"; D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_cprobe CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1|1|$D 1 $S
dp2|2|$D 2 $S
dp2_ep2|2|$D 2 $E 2 $S" $MX cp7_base
for cfg in cp2 cp2_generic cp2_allgather cp2_allgather_generic; do
TITAN=/tmp/wt_cprobe CFG=kimi_k3_debugmodel_$cfg BATCH="$B" CELLS="$cfg|2|$D 1" $MX cp7_$cfg
done
TITAN=/tmp/wt_cprobe CFG=kimi_k3_debugmodel_cp2 BATCH="$B" CELLS="cp4|4|$D 1 $C 4
cp8|8|$D 1 $C 8
dp2_cp2|4|$D 2
dp2_ep2_cp2|4|$D 2 $E 2
dp2_cp4|8|$D 2 $C 4
dp2_ep2_cp4|8|$D 2 $E 2 $C 4
dp4_cp2|8|$D 4
dp4_ep2_cp2|8|$D 4 $E 2
dp4_ep4_cp2|8|$D 4 $E 4" $MX cp7_mix
echo "CP7 MATRIX DONE"
