# Source commits (fork torchtitan `attention_residual_dev`)

| commit | scope |
|---|---|
| pending (working tree on top of `a6f582b3b` when this kit was written) | `clip_grad_norm_`: compute the total norm in float32. **`torchtitan/distributed/utils.py` only**, +54/−3, no other file touched. `grad_norm_fp32_PR26.patch` in this folder is that hunk verbatim. |

## Discovery phase

Found 2026-08-09 while making a multimodal PP layout (vision tower on its own stage) agree
with the layout that keeps the tower inline. From a shared seed checkpoint the two layouts
produced:

* identical step-1 loss, to every printed digit;
* identical gradients — all 788 per-shard gradient norms matched, max relative difference
  `0.000e+00`;
* but `grad_norm` 10.008054 against 9.951641.

Summing those 788 shard norms in float32 gives **9.989287 for both layouts**, so neither
reported number was right — they were wrong in opposite directions, by +0.19% and −0.38%.
With `max_norm=1.0` against a true norm near 10, clipping fires every step, so the clip
factor differed by 0.566% and the two layouts took different-sized steps from identical
gradients.

Full write-up, including the traceback that cleared the model code:
`phase13_k3like_48b_posttrain/DEP_ALIGNMENT_2026-08-09.md`.

## Cherry-pick recipe

Self-contained and touches no model code, so a hand-port is simplest:

    git checkout -b grad-norm-fp32 upstream/main
    git apply Raising_PRs/PR26_torchtitan_grad_norm_low_precision/grad_norm_fp32_PR26.patch

If it conflicts, the change is three mechanical edits:

1. Add `_get_total_norm_fp32` immediately above `clip_grad_norm_`.
2. Replace `torch.nn.utils.get_total_norm(` with `_get_total_norm_fp32(` at all three call
   sites — one in `clip_grad_norm_`, two in `_clip_grad_norm_with_ep`.
3. Nothing else. `Iterable` is already imported in that file upstream.

The patch is clean: only the helper and the three call-site replacements. The fork's
`pp_mesh` dtype fix (see below) was committed earlier and is not in this diff.

One thing to check when hand-porting, because I got it wrong the first time: `clip_grad_norm_`
carries an `@torch.no_grad()` decorator. Inserting the helper immediately above the `def` puts
it under that decorator, which then applies to the helper and leaves `clip_grad_norm_`
undecorated. Both functions need their own.

## Relationship to the other fix in the same function

The fork also carries a `pp_mesh` dtype fix in `clip_grad_norm_` /
`_clip_grad_norm_with_ep`. `get_total_norm` returns a CPU float32 `tensor(0.)` for an empty
gradient list, so a PP rank contributing no gradients to one group reached the `all_reduce`
in float32 while its peers carried bfloat16 — and **NCCL returns garbage on a dtype mismatch
instead of raising**. The observed symptom was one side of the pipeline reporting a plausible
norm (its own shard's sum only) and the other reporting NaN, while every individual gradient
was finite.

That is a separate, independently filable patch in the same function. Filing both together is
reasonable — same reviewer, same code, same family of defect — but neither depends on the
other, and either can go first.

The float32-norm patch does **not** subsume it: the empty-list case still returns a CPU
tensor whose device and dtype have to be normalised before the collective.

## Not for upstream

Nothing else from the fork. The kimi_k3 model code, the DEP probes
(`matrix_scripts/dep_grad_multiset_probe.py`) and the matrix scripts stay downstream. The
probe is worth mentioning in review only if a maintainer asks how the gradients were shown
identical.
