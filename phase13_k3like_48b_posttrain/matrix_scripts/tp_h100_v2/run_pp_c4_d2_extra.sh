#!/bin/bash
# Follow-up to run_pp_c4.sh's 2048-token stream: the accumulation-matched reference (dp2, fp32 no-sync
# accumulation, as the pipeline cells accumulate), a dp2 noise-floor row (micro-batch order reversed), and
# d2_pp2 again on a copy of its own inductor cache (is its steps 7-12 excursion deterministic?). Same OUT.
set -u; export CFG=kimi_k3_debugmodel_c4; . "$(dirname "$0")/common.sh"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
( export NOSYNC_GA=1; cell c4_d2_dp2_ns $TT 0,1 2 c4_2048 100 $B8 $D 2 ) &
( export MB_REVERSE=1; cell c4_d2_dp2_rev $TT 2,3 2 c4_2048 100 $B8 $D 2 ) &
wait
rm -rf $OUT/ind_c4_d2_pp2_again $OUT/tri_c4_d2_pp2_again
cp -r $OUT/ind_c4_d2_pp2 $OUT/ind_c4_d2_pp2_again; cp -r $OUT/tri_c4_d2_pp2 $OUT/tri_c4_d2_pp2_again
cell c4_d2_pp2_again $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P
export TABLE_STEPS="1 5 7 8 10 12 20 50 100"
echo; echo '# c4, 2048 tokens per step, reference dp2'
table c4_d2_dp2 c4_d2_dp2_rev c4_d2_dp2_ns c4_d2_ep2 c4_d2_pp2 c4_d2_pp2_again c4_d2_vp2c c4_d2_vp2n
echo; echo '# the same, reference dp2 no-sync (fp32 accumulation, as the pipeline cells)'
table c4_d2_dp2_ns c4_d2_pp2 c4_d2_pp2_again c4_d2_vp2c c4_d2_vp2n
echo RUN-PP-C4-D2-EXTRA-DONE
