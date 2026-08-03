# Evidence invalidation: routed experts were never initialized (2026-07-27)

## What was found

`KimiK3Model.init_weights` never initialized the routed-expert parameters.
Two independent stale assumptions in the same code block:

```python
if cls_name == "GroupedExperts":          # class-name equality
    for name in ("w1", "w2", "w3"):       # pre-rename parameter names
```

* Upstream renamed the expert parameters to shape-suffixed `w1_EFD` / `w2_EDF` /
  `w3_EFD`, so `getattr(m, "w1")` returned `None` and the loop was a no-op even
  for the core class.
* K3's routed experts became a `KimiSiTUGroupedExperts` subclass (2026-07-27,
  SiTU-GLU), so the class-name equality stopped matching at all.

torchtitan's flow is meta-build -> FSDP wrap -> `to_empty` -> `init_weights`, so
`init_weights` is the *only* thing that gives these parameters values. The
routed experts therefore ran on whatever `to_empty` left on the device.

Fixed in titan `3a7cd765`; guarded by
`torchtitan/experiments/kimi_k3/tests/test_expert_init.py` (5 tests, including
an end-to-end "zeroing every expert must change the output" check).

## Why it was silent

The model trains to a completely plausible loss curve without the routed
experts: the dense FFN on layer 0, the two full-width shared experts, and the
latent projections carry the FFN path on their own. Nothing errors, no
gradient is NaN, and grad_norm stays in a normal range.

Measured at dp2 on `kimi_k3_mini_block_attn_res`, seed 42 deterministic
(`moe_connected_probe.py`):

| configuration | loss | routed-expert grad-norm |
| --- | --- | --- |
| before the fix | 5.666818 | 0.0 |
| **every expert weight explicitly zeroed** | **5.666818** | 0.0 |
| after the fix | 5.611669 | 1.42e+01 |
| after the fix, experts zeroed | 5.666818 | 0.0 |

The pre-fix loss is *bit-identical* to the deliberately-zeroed control. That is
the cleanest possible statement of the problem: the routed experts contributed
exactly nothing.

Trainer-level effect on the same 3-step smoke:

| flavor | before | after |
| --- | --- | --- |
| `kimi_k3_mini_block_attn_res` | 7.71445 / 7.61991 / 7.15075 | 7.75113 / 7.68023 / 7.59077 |
| `kimi_k3_mini_qat_mxfp4` | 7.71445 / 7.61991 / 7.15075 | 7.72159 / 7.65503 / 7.55739 |

Note the pre-fix loss fell *faster* (7.71 -> 7.15 vs 7.75 -> 7.59). A smaller
effective model overfits a 2-sequence batch more quickly, so the broken
configuration looked better on the only metric being watched.

## A second result this invalidates

The first MXFP4-QAT A/B looked like a pass for the wrong reason. Base and QAT
losses were bit-identical (7.71445), which I flagged as the vacuous-A/B
signature; the investigation that followed found the init bug. But the earlier
*offline* comparison had shown QAT changing the loss while the base did not --
apparently confirming QAT worked. It did not: MX-quantizing an all-zero tensor
yields garbage rather than zeros (the block scale is derived from `amax = 0`),
so the fake-quant was accidentally *supplying* the weights that init had
skipped. With live experts the A/B is genuine: 7.75113 (base) vs 7.72159 (QAT).

Same lesson as the FlashKDA vacuous A/B recorded in
`K3_RECONCILIATION_2026-07-27.md`: an A/B that reports "no difference" and an
A/B that reports "a difference" are both worthless until you have a control
proving the harness can tell the two arms apart. Here the control was "zero the
thing and check the output moves", and it costs one extra forward pass.

## Which recorded numbers are affected

Anything with routed experts, produced after the upstream merge that introduced
the parameter rename (titan `469577cdf`, 2026-07-17). Concretely, every MoE
loss in:

* the fsdp2 baseline `7.5695` and the whole packed-MXFP4 x TP matrix
  (`PACKED_TP_VERIFICATION_2026-07-25.md`, `run_packed_tp_matrix.sh`)
* the 8-GPU gap matrices (`run_remaining_8gpu_gaps.sh`, `run_gaps_redo.sh`)
* every EP smoke, and the 4D/5D mesh combination losses
* the FSDP gradient-division measurements' absolute loss values

**What survives.** The invalidation is about *loss values*, not about the
mechanisms those runs were built to test. Each of those runs was a structural
verification -- does this parallelism combination compose, does it run
rank-identically, does the packed-MXFP4 base survive a TP shard, is the
gradient summed rather than averaged over the dp axes. Those conclusions rest
on crashes-vs-no-crash, on rank-identity, and on directly measured
`param.grad` values, none of which depend on the routed experts being alive.
The DSv3 control legs used upstream models with upstream's declarative init and
were never affected at all.

So: the parallelism findings stand, the loss numbers do not. Any comparison
that reads a loss value from before 2026-07-27 against one from after it is
comparing two different models.

## Follow-up

Migrate `init_weights` to upstream's declarative per-parameter init map (see
`models/deepseek_v3/__init__.py::_depth_experts_init`, which maps `w1_EFD` /
`w2_EDF` / `w3_EFD` to `trunc_normal_` with depth-scaled std). Two reasons
beyond robustness: a missing entry is visible in one table rather than hidden
in a walk, and our pass currently lacks the depth scaling upstream applies to
`w2`/`w3`. Tracked as a TODO in `model.py`.
