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
