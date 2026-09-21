#!/bin/bash
# After phase C of run_pp_h200.sh lands, stop the driver before phase D (the bf16-norm appendix is not in the body),
# then the reruns: the two references with the logged-loss equalization and pp2 (port collision), then every cell
# that is not bitwise with its reference once more on a copy of its own finished cache (<cell>_w). Progress lines
# go to the run log so the same watch sees them.
set -u
OUT=/workspace/results/pp_r3; K=/workspace/kit; LOG=/workspace/results/pp_r3_run.log
until grep -q 'd2_vp2n rc=' $LOG; do sleep 3; done
pkill -f run_pp_h200.sh; sleep 3; pkill -f 'torchtitan.train'
while pgrep -f 'torchtitan.train' > /dev/null; do sleep 2; done
echo "STOPPED-AFTER-C-DONE $(date -u +%T)" >> $LOG
cd $OUT
for c in dp1_ns dp1_ns_cold f32_dp1_ns f32_dp1_ns_cold; do mv $c.log ${c}_seqsum.log; done
cp -r ind_dp1_ns ind_dp1_ns_seqsum; cp -r ind_f32_dp1_ns ind_f32_dp1_ns_seqsum
rm -rf bf_dp1_ns_cold ind_bf_dp1_ns_cold tri_bf_dp1_ns_cold bf_dp1_ns_cold.log
echo "== R1: references with the logged-loss equalization, pp2 $(date -u +%T)" >> $LOG
GPUS=0 PORT=41000 bash $K/rerun_cells.sh dp1_ns >> $LOG 2>&1 &
GPUS=1 PORT=42000 bash $K/rerun_cells.sh f32_dp1_ns >> $LOG 2>&1 &
GPUS=2,3 PORT=43000 bash $K/rerun_cells.sh pp2 >> $LOG 2>&1 &
wait
departs() {  # departs <ref> <cell>: 0 when the cell's grad norm is not bitwise with the reference on all steps
  python3 $K/tables.py $OUT $1 $2 2>/dev/null | tail -n 1 | grep -q ', 100/100 |' && return 1; return 0
}
A="warm:dp1 warm:vp2c warm:pp2vp4c warm:f32_vp2n"; B="warm:dp1_rev warm:f32_vp2c"; C="warm:pp4vp2c warm:pp4vp4c warm:f32_pp4vp4n warm:f32_pp4vp4c"
departs d2_dp2_ns d2_dp2 && A="$A warm:d2_dp2"
departs d2_dp2_ns d2_ep2 && B="$B warm:d2_ep2"
for c in d2_pp2 d2_vp2c d2_vp2n; do departs d2_dp2_ns $c && C="$C warm:$c"; done
echo "== R2: warm self-reruns: [$A] [$B] then [$C] $(date -u +%T)" >> $LOG
GPUS=0,1 PORT=44000 bash $K/rerun_cells.sh $A >> $LOG 2>&1 &
GPUS=2,3 PORT=45000 bash $K/rerun_cells.sh $B >> $LOG 2>&1 &
wait
PORT=46000 bash $K/rerun_cells.sh $C >> $LOG 2>&1
export TABLE_STEPS="1 10 20 50 100"
{
echo "# A: c4, 1024 tokens per step (4 x 256), bf16, fp32 total grad norm; reference dp1 with matched accumulation (warm)"; python3 $K/tables.py $OUT dp1_ns dp1_ns_cold dp1_ns_seqsum dp1 dp1_w dp1_rev dp1_rev_w pp2 pp4 vp2n vp2c vp2c_w pp2vp4n pp2vp4c pp2vp4c_w pp4vp2n pp4vp2c pp4vp2c_w pp4vp4n pp4vp4c pp4vp4c_w
echo; echo "# B: fp32 parameters, compute and reduction (FP32_PROBE), cache on and off"; python3 $K/tables.py $OUT f32_dp1_ns f32_dp1_ns_cold f32_dp1_ns_seqsum f32_dp1 f32_vp2n f32_vp2n_w f32_vp2c f32_vp2c_w f32_pp4vp4n f32_pp4vp4n_w f32_pp4vp4c f32_pp4vp4c_w
echo; echo "# C: c4, 2048 tokens per step (4 x 256 per rank), dp2, bf16, fp32 norm"; python3 $K/tables.py $OUT d2_dp2_ns d2_dp2_ns_cold d2_dp2 d2_dp2_w d2_ep2 d2_ep2_w d2_pp2 d2_pp2_w d2_vp2n d2_vp2n_w d2_vp2c d2_vp2c_w
} > $OUT/tables_r.md 2>&1
echo "RERUN-ALL-DONE $(date -u +%T)" >> $LOG
