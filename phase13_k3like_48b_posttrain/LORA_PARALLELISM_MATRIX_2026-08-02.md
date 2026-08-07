# LoRA across the parallelism combinations

Warm checkpoint (LoRA's B is zero at init, so a cold check measures an inert
adapter and reports it clean -- that is how the o_proj defect survived one), one
step compared, each leg against the reference sharing its accumulation structure.
364 parameters carry gradients under LoRA; the base is frozen, so the trainable
set is adapters, norms and the AttnRes pseudo-queries.

  leg                  max      median   LoRAmax   max over params >1% of norm
  ref / fsdp2       0.00000    0.00000   0.00000   --
  ep2_fsdp2         0.00000    0.00000   0.00000   --
  pp2               0.28647    0.01456   0.20355   0.10513  mlp_res_proj
  cp2               0.20148    0.01609   0.20148   0.14971  up.lora_b
  tp2               1.33869    0.01890   1.33869   0.16515  attn_res_proj
  pp2_cp2           0.16856    0.01442   0.13217   0.16236  mlp_res_proj
  fsdp2_pp2         0.32768    0.01455   0.23451   0.28263  attn_res_proj
  fsdp2_cp2         0.35406    0.01395   0.11670   0.29417  attn_res_proj
  fsdp2_pp2_cp2     0.32233    0.01568   0.24654   0.27817  attn_res_proj
  ep2_fsdp2_pp2     0.32768    0.01455   0.23451   --
  ep2_fsdp2_cp2     0.35406    0.01395   0.11670   --

## What is clean

**FSDP alone is exact** -- 0.00000 on every parameter. So is EP+FSDP. That also
validates the instrument: LoRA *can* measure exact here, so the other numbers are
not an artifact of the setup.

**EP contributes nothing.** ep2_fsdp2_pp2 equals fsdp2_pp2 and ep2_fsdp2_cp2
equals fsdp2_cp2 to five decimals. Turning EP on changes no digit.

## What is not

**TP is the known o_proj.lora_b defect** (LORA_TP_DEFECT), 1.33869 unweighted.

**PP and CP each carry a real deviation under LoRA that they do not carry for
full-parameter training.** Weighted to parameters holding more than 1% of the
gradient norm, PP is 0.105, CP is 0.150, and the FSDP combinations are 0.28-0.29.
For the full-parameter model the same axes measured 0.00000 (PP, per-parameter
over 548 params) and ~0.007 (CP). So this is specific to the LoRA path.

The worst-weighted parameter is almost always an AttnRes pseudo-query
(attn_res_proj / mlp_res_proj). Under LoRA the base is frozen, so those
projections are a much larger share of the trainable signal than they are in
full-parameter training, and they are zero-initialized, which is where the
AttnRes softmax cancellation is worst. Whether that fully accounts for 28% has
NOT been established -- the same reasoning was measured and rejected once already
for the MoE residual, where the cancellation turned out to be 15x rather than the
1e5 the story needed.

## Standing

LoRA usable: FSDP, EP+FSDP. Not usable without further work: TP (known defect),
PP and CP (unexplained, this document). The replacement implementation
(LORA_CONVERTER_BLOCKER) has to clear these numbers, not just the TP one.

## Rerun 2026-08-07, after the o_proj fix

Same instrument, same flavor (`kimi_k3_mini_qlora`), same warm-checkpoint
protocol, 364 parameters carrying gradients.

| leg | 2026-08-02 | 2026-08-07 |
|---|---|---|
| ref / fsdp2 | 0.00000 | **0.00000** |
| ep2_fsdp2 | 0.00000 | **0.00000** |
| **tp2** | **1.33869** | **0.23221** |
| pp2 | 0.28647 | 0.23654 |
| cp2 | 0.20148 | 0.31179 |
| pp2_cp2 | 0.16856 | 0.21852 |
| fsdp2_pp2 | 0.32768 | 0.21488 |
| fsdp2_cp2 | 0.35406 | 0.26581 |
| fsdp2_pp2_cp2 | 0.32233 | 0.26454 |
| ep2_fsdp2_pp2 | 0.32768 | 0.21488 |
| ep2_fsdp2_cp2 | 0.35406 | 0.26581 |

**TP is no longer an outlier.** 1.33869 to 0.23221 -- from roughly six times the
PP/CP background to inside it. That is the `o_proj` Rowwise-bypass defect
(LORA_TP_2026-08-05, "RESOLVED") leaving the measurement. FSDP and EP+FSDP are
still exact, so the instrument is still validated by its own control.

**PP and CP are unchanged in character and still unexplained.** 0.21-0.31, and the
worst-weighted parameter is still an AttnRes pseudo-query norm. The fixed defect
cannot account for them: that branch requires a base weight sharded on the
contracted axis, which only exists under TP, so it never fires under pure PP or
pure CP.

### One thing this rerun surfaced that is NOT explained

`cp2` MOVED, 0.20148 to 0.31179, in a direction the fix cannot cause. Checked
whether that is run-to-run variation, because assuming either answer would be
guessing: two independent repeats of the `cp2` leg against the same warm checkpoint
give **364/364 bit-identical gradients** and max = 0.31179 exactly. So the
measurement is deterministic and the movement is a real consequence of some change
landed between 2026-08-02 and now -- and `kimi_k3_mini_qlora` is text-only, so the
ViT CP image-sharding work is not the candidate it would otherwise be.

Not attributed. Identifying it is a bisect over five days of commits with this
probe as the oracle, and it is the next thing this line of work owes.

### A collector bug in the runner, now fixed

`run_overnight_2026_08_02.sh` ended with

    python3 - "$OUT" 2>&1 | tee -a "$LOG" <<'PY' ... PY

A heredoc after a pipeline attaches to the LAST command, so `tee` consumed the
script and appended its source to the log while `python3 -` read nothing. The
comparison silently produced no table -- the failure that looks like a completed
run. The numbers above were produced by running the comparison over the dumps
separately, which is also why `collect13.sh` is kept out of its runner.
