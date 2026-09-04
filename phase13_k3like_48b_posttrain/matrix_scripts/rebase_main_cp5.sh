#!/bin/bash
# cp_review5 (CP on main + the CP kernel stack + 4446): the CP PR's own cells on the multimodal
# flavor -- dp1, dp2, cp2 with the four MLA kernels, cp4, dp2 x cp2, and dp2 x ep2 x cp2 as the
# optional EP cell. The debug flavor pins partial_dtensor, so the base cells pass spmd_types.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel_mm
S="--parallelism.spmd_backend spmd_types"; D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_cprun5 CFG=kimi_k3_debugmodel_mm BATCH="$B" CELLS="dp1|1|$D 1 $S
dp2|2|$D 2 $S" $MX cp5_base
for cfg in cp2 cp2_generic cp2_allgather cp2_allgather_generic; do
TITAN=/tmp/wt_cprun5 CFG=kimi_k3_debugmodel_$cfg BATCH="$B" CELLS="$cfg|2|$D 1" $MX cp5_$cfg
done
TITAN=/tmp/wt_cprun5 CFG=kimi_k3_debugmodel_cp2 BATCH="$B" CELLS="cp4|4|$D 1 $C 4
dp2_cp2|4|$D 2
dp2_ep2_cp2|4|$D 2 $E 2" $MX cp5_mix
echo "CP5 MATRICES DONE"
#!/bin/bash
# cp8 on cp_review5: the packed Ulysses default (kimi_k3_debugmodel_cp2 with the degree overridden), 8 GPUs.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel_mm
D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_cprun5 CFG=kimi_k3_debugmodel_cp2 BATCH="$B" CELLS="cp8|8|$D 1 $C 8" $MX cp5_cp8
echo "CP8 DONE"
