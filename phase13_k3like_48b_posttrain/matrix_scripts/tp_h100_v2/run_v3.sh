#!/bin/bash
# TP/SP #4499 round 3 on 4 x H100: tpsp_review4 (1dec3ee17) against main d34a13fdf. Reruns the body's
# tables after the rebase (#4535 moved the tp=1 reference), adds the routed_down path this round changed
# (dp2 x ep2 x tp2 without SP), and smokes the b200 cell kimi_k3_debugmodel_mm through its recipe.
# Set TT (branch tree), TT_PARENT (main tree), OUT.
set -u; . "$(dirname "$0")/common.sh"
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
B2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
seed bf16 $B
seed bf16_dp2 $B2
# dp1 stream: reference tp=1 on main
cell tp1_parent $TT_PARENT 0   1 bf16 100 $B $D 1 &
cell tp1        $TT        1   1 bf16 100 $B $D 1 &
cell tp2_sp     $TT        2,3 2 bf16 100 $B $D 1 $T 2 &
wait
cell tp2_nosp   $TT        0,1 2 bf16 100 $B $D 1 $T 2 $NOSP &
cell tp1_again  $TT        2   1 bf16 100 $B $D 1 &
wait
# dp2 stream: reference dp2 on this branch (a second dp rank reads other samples)
cell dp2              $TT 0,1     2 bf16_dp2 100 $B2 $D 2 &
cell dp2_ep2          $TT 2,3     2 bf16_dp2 100 $B2 $D 2 $E 2 &
wait
cell dp2_tp2          $TT 0,1,2,3 4 bf16_dp2 100 $B2 $D 2 $T 2
cell dp2_ep2_tp2      $TT 0,1,2,3 4 bf16_dp2 100 $B2 $D 2 $E 2 $T 2
cell dp2_ep2_tp2_nosp $TT 0,1,2,3 4 bf16_dp2 100 $B2 $D 2 $E 2 $T 2 $NOSP
# Kimi K2.5 at tp=1, this branch against main; no seed checkpoint on K2.5, both start from --debug.seed
K3_COMMON=$COMMON
COMMON=${K3_COMMON/--module kimi_k3 --config $CFG/--module kimi_k2_7 --config kimi_k2_5_debugmodel}
[[ $COMMON == *kimi_k2_5_debugmodel* ]] || { echo "K2.5 command substitution failed"; exit 1; }
cell k27_dp2_parent $TT_PARENT 0,1 2 k27 100 $D 2 &
cell k27_dp2        $TT        2,3 2 k27 100 $D 2 &
wait
# Type-checking smokes through the b200 recipe (type checking on, AC off), 3 steps, fresh init
COMMON=${K3_COMMON/--module kimi_k3 --config $CFG/--module torchtitan_recipes.tests.b200 --config kimi_k3_debugmodel_mm}
[[ $COMMON == *kimi_k3_debugmodel_mm* ]] || { echo "recipe command substitution failed"; exit 1; }
cell tc_mm      $TT 0,1,2,3 4 none 3
cell tc_mm_nosp $TT 0,1,2,3 4 none 3 $NOSP
cell tc_tp2     $TT 0,1     2 none 3 $D 1 $T 2 $E 1
COMMON=$K3_COMMON
echo
echo '# dp1, 256 tokens per step (reference: tp=1 on main d34a13fdf; tp1 and tp1_again must be bitwise with it)'
table tp1_parent tp1 tp1_again tp2_sp tp2_nosp
echo
echo '# dp2 stream, 512 tokens per step (256 per rank; reference: dp2 on this branch)'
table dp2 dp2_ep2 dp2_tp2 dp2_ep2_tp2 dp2_ep2_tp2_nosp
echo
echo '# Kimi K2.5, dp2, tp=1, 4096 tokens per step (reference: main)'
table k27_dp2_parent k27_dp2
echo
for c in tc_mm tc_mm_nosp tc_tp2; do echo "$c: $(grep -a -o 'step: *[0-9]*' $OUT/$c.log | sort -u | wc -l) distinct steps, $(grep -a -ciE 'Traceback' $OUT/$c.log) tracebacks"; done
echo RUN-V3-DONE
