# Where the 13-leg matrix drift came from

Re-ran `BASELINE_MATRIX_2026-08-03.md`'s 13 legs at three code states with one
script (`run_full13_matrix.sh`, `TITAN=` selects the tree) and the same fixture:
`diag_4l_moe_depth`, 3 steps, seed 42, deterministic, global batch 8.

    recorded  = the numbers in BASELINE_MATRIX_2026-08-03.md
    pre-alpha = 22d36d8ed (b0484a21c^, the commit before "compute alpha in FP32")
    post-alpha= b0484a21c
    HEAD      = 32f5398b9

## Result 1: alpha-FP32 accounts for every HEAD-vs-pre-alpha difference

post-alpha and HEAD are **bit-identical on all 13 legs**. The four commits after
it -- the per-instance fla shims, the empty-block guard, the PP8/VP4 flavor, and
the KCP conv halo -- move nothing on this matrix, which matches their scope
(this flavor is MLA+MoE, so no KDA and no KCP path).

## Result 2: alpha bites exactly where bf16 enters, and nowhere else

Which legs alpha-FP32 changed, sorted by their mesh:

| leg | dp_shard | CP | changed by alpha? |
|---|---|---|---|
| dp1, tp2, pp2 | 1 | no | **no -- bit-identical** |
| cp2 | 1 | yes | yes |
| tp2_pp2_cp2 | 1 | yes | **FAIL -> PASS** |
| fsdp2, ep2_fsdp2, and every 3-of-4 leg | 2 | either | yes |

The rule is clean: unchanged iff `dp_shard == 1 AND no CP`. That is the set of
configurations where nothing casts the parameters down -- FSDP2's mixed
precision is what puts them in bf16, and CP's all-to-all is the other place the
model dtype shows up. Computing alpha in FP32 is a no-op when the surrounding
compute is already FP32, which is why single-GPU, TP-only and PP-only are
untouched.

So the answer to "which parallelism caused it" is: **FSDP2 (via mixed precision)
and CP (via the all-to-all dtype)** -- and not as a defect. The change was
deliberate, it matches the release, and it turned `tp2_pp2_cp2` from a hard
failure into a passing leg.

## Result 3: nothing here is nondeterministic

Every leg reproduced bit-exactly between the post-alpha run and the HEAD run --
two independent 8-GPU runs, same numbers to the last printed digit, including
the FSDP and CP legs. The earlier guess that the FSDP/CP spread might be
run-to-run reduction-order noise is wrong; these configurations are
reproducible.

## Result 4: the recorded doc numbers are stale, and NOT because of our commits

`dp1` and `tp2` are bit-identical at pre-alpha and at HEAD, so no commit in this
window touched them -- yet neither matches the recorded value. Bisecting `dp1`
back across the whole window:

    00fbd3a30  7.72352 7.14592 6.22468
    43dab399b  7.72352 7.14592 6.22468   (the upstream merge)
    75500d2fa  7.72352 7.14592 6.22468   (experiments/ -> models/)
    59dffc845  7.72352 7.14592 6.22468
    ccc17cf56  7.72352 7.14592 6.22468
    8968d92ae  7.72352 7.14592 6.22468

Identical at all six, and the recorded value is 7.72376 7.14969 6.22981. The
divergence therefore predates every commit tested, so it is environmental or a
fixture detail, not a regression. `TORCH_UPGRADE_2026-08-03.md` reports its own
matrix as bit-identical across the torch bump, so that is not obviously the
cause either. **This is unresolved**: pinning it down needs the pre-upgrade
environment, which is gone. The practical consequence is that the recorded
table should not be used as a regression fixture any more.

## Current numbers, to be used as the new fixture

    dp1                    7.72352 7.14592 6.22468
    fsdp2                  7.67804 7.27652 6.28913
    pp2                    7.66117 7.04581 6.12275
    cp2                    7.69538 7.21654 6.26876
    tp2                    7.70012 7.04306 6.15943

    fsdp2_tp2_pp2          7.71112 7.23708 6.27646
    fsdp2_tp2_cp2          7.76412 7.31589 6.38356
    tp2_pp2_cp2            7.72024 7.15531 6.29512
    fsdp2_pp2_cp2          7.70294 7.23875 6.33552

    ep2_fsdp2              7.67804 7.27523 6.28938
    ep2_fsdp2_tp2_pp2      7.73794 7.21240 6.31050
    ep2_fsdp2_tp2_cp2      7.71993 7.22795 6.37054
    ep2_fsdp2_pp2_cp2      7.75901 7.25013 6.52894

13/13 pass. Confirmed reproducible: the post-alpha and HEAD runs produced these
same values independently.
