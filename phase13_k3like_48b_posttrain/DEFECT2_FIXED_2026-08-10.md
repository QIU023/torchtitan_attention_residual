# Defect 2 fixed: the multimodal matrix is 18/18 with DEP and dynamic CP on

Three TP+CP cells had failed in every published matrix, and the workaround was to force
`KIMI_VIT_DYNAMIC_CP=0` on them. They now pass, and the fifteen cells that already passed
are byte-identical to the pre-fix run.

    NotImplementedError: Operator c10d.allgather_.default does not have a sharding
    strategy registered.

## The recorded cause was too broad

It was carried as "gather-KV on DTensors under vision TP; needs the `wo` local/DTensor
contract settled first". The real trigger is narrower: **replicated vision attention**.

`parallelize.py` head-shards vision attention only when the head count divides the TP
ranks; otherwise it warns and leaves attention replicated. The head-sharded branch already
drops q/k/v to local tensors before the gather -- which is why it worked -- and the
replicated branch did not, so the gather received DTensors.

`kimi_k3_debugmodel_report_arch` has **3** vision heads against tp2, so it always took the
replicated branch. The test that settled it: `kimi_k3_debugmodel_report_arch_vit4h`, the
same flavor with 4 heads, ran `fsdp2_tp2_cp2` **unmodified** for 3 steps. Same code, same
cell, head count the only difference.

That reframing is what made the fix small. Nothing about `wo`'s contract had to be
renegotiated -- in the replicated branch `wo` is not in the TP plan at all, so
`distribute_module` leaves it replicated and it simply wants a replicated DTensor back.

## The fix

In `MoonViTEncoderLayer._attend`, inside the CP-plan branch: take replicated q/k/v to
local, gather, then re-wrap the attention output as `Replicate` before `wo`. Both
conversions are exact rather than coercions, since the local shard of a replicated DTensor
IS the full tensor. Placements are asserted, not assumed -- a non-`Replicate` placement
raises rather than being silently reinterpreted.

**The load-bearing detail is `grad_placements=[Replicate()]`, not `Partial()`.** Every TP
rank runs the same full-head attention and receives the same full gradient, so a `Partial`
declaration would be summed over the TP axis and scale the gradient by `tp_size`. The
head-sharded branch above uses `Partial` correctly, for the opposite reason: its head
slices are disjoint, so each rank's gradient is an additive part of one whole.

## Evidence

**The three cells run and converge**, 10 steps, DEP and dynamic CP both on:

| cell | step 1 | step 10 |
|---|---|---|
| `fsdp2_tp2_cp2` | 12.08587 | 9.93480 |
| `tp2_pp2_cp2` | 12.05198 | 9.74631 |
| `ep2_fsdp2_tp2_cp2` | 12.04774 | 9.95196 |

They land where their topology says they should: the two without PP at 9.93-9.95 next to
the other non-PP cells' 9.90-9.93, and `tp2_pp2_cp2` at 9.7463 next to the other PP cells'
9.742-9.753.

**Nothing else moved.** All 10 other cells of the 13-cell half and all 5 max-degree cells
are byte-identical to the pre-fix table (`diff` over the collected rows, excluding the
three target cells, is empty).

**The gradient placement, on a flavor where both branches are reachable.** 4-head tower,
tp2 x cp2, dynamic CP on:

| arm | step-1 grad_norm |
|---|---|
| head-sharded (pre-existing path) | 12.8518 |
| replicated (this fix's path, `KIMI_VIT_TP_HEADS=0`) | 12.7671 |
| tp1 reference, no TP at all | 12.6299 |

Within 1.8% of each other, which is bf16 summation order across different shardings. A
`Partial` declaration would have shown roughly double.

**Negative control, run rather than argued.** Flipping the declaration to `Partial` makes
the new unit test fail on the GRADIENT assertion with `Greatest relative difference: 1.0`
-- exactly a factor of two -- while the OUTPUT assertion still passes. So an output-only
test would have accepted the wrong placement, which is why the test asserts the gradient.

## Test

`tests/test_moonvit_dynamic_cp_replicated_tp.py`, 2 gloo ranks. It replicates the layer
over a TP mesh so `wqkv`'s output is a `DTensor(Replicate)` and `_tp_head_slice` is None --
exactly what `parallelize.py` leaves behind -- and compares output and `wqkv` gradient
against the plain-tensor path.

The gather itself is replaced by a local stand-in, and that is a real limitation worth
stating: on gloo, `dist_nn.all_gather`'s backward cannot run on a process subgroup at all,
because its scatter fallback passes a group-local index where a global rank is expected and
any subgroup not starting at global rank 0 raises. NCCL takes the `all_to_all` branch and
does not hit it. So a CPU test of tp2 x cp2 end to end is not expressible; the end-to-end
case is covered by the three matrix cells above. The first version of this test did attempt
it and failed inside torch, not inside our code.

## Status of the matrix

`kimi_k3_debugmodel_report_arch`, DEP=1, dynamic CP=1, eager, 10 steps: **18 of 18 cells
pass.** `KIMI_VIT_DYNAMIC_CP=0` is no longer needed anywhere, and the note in
`MATRIX_18_CORRECTED_2026-08-09.md` and `MATRIX_DEP_DYNCP_2026-08-10.md` that the three
cells need it is superseded.

The PP cells still differ from the `MATRIX_18_SDPA` baseline by 7.5e-06 to 2.6e-03, which
is DEP's cold-initialization effect and is unrelated to this fix -- see
`MATRIX_DEP_DYNCP_2026-08-10.md`, where the `pp2` DEP-off control reproduces the baseline
exactly.

## Repro

    # the fix, all three cells
    KIMI_VIT_DEP=1 KIMI_VIT_DYNAMIC_CP=1 OUT=/workspace/mx_d2fix STEPS=10 \
      FLAVOR=kimi_k3_debugmodel_report_arch bash matrix_scripts/run13_flav.sh
    # the head-count control: same cell, 4 heads, passes without the fix
    --config kimi_k3_debugmodel_report_arch_vit4h
    # the placement A/B: add KIMI_VIT_TP_HEADS=0 to force the replicated branch
