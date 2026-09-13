#!/bin/bash
# pp4 x vp4 (16 stages, Interleaved1F1B) on run_pp_c4.sh's 1024-token stream, all four GPUs, same seed. The tree
# needs pp_stages_per_rank.patch (PP_STAGES_PER_RANK; no layers_per_stage splits the 35 units into 16). dp2 x pp4
# needs eight GPUs and is not here. Same OUT as run_pp_c4.sh.
set -u; export CFG=kimi_k3_debugmodel_c4 PP_STAGES_PER_RANK=4; . "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
seed c4_1024 $B4
cell c4_pp4vp4c $TT 0,1,2,3 4 c4_1024 100 $B4 $D 1 $P4
( naive_common; cell c4_pp4vp4n $TT 0,1,2,3 4 c4_1024 100 $B4 $D 1 $P4 )
export TABLE_STEPS="1 10 20 50 100"
echo; echo '# c4, 1024 tokens per step (4 x 256), reference dp1'
table c4_dp1 c4_dp1_rev c4_dp1_ns c4_pp2 c4_vp2c c4_vp2n c4_pp4vp4c c4_pp4vp4n
echo; echo '# the same, reference dp1 no-sync (fp32 accumulation, as the pipeline cells)'
table c4_dp1_ns c4_pp2 c4_vp2c c4_vp2n c4_pp4vp4c c4_pp4vp4n
echo RUN-PP-C4-PP4VP4-DONE
