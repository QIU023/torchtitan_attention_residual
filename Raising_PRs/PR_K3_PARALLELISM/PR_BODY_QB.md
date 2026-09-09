# PR title: [Kimi K3] Quantile balancing for the MoE router bias

PR 4412. Content: `qb_review4` = `743cefe6a`, three commits on upstream/main `65ba8a697`; `git diff upstream/main` is exactly `torchtitan/components/quantile_balance.py` (+407), `tests/unit_tests/cpu/test_quantile_balance.py` (+273) and `torchtitan/models/kimi_k3/config_registry.py` (+16). Sync: `git push origin qb_review4:k3_qb --force-with-lease`, drop the "[DO NOT review, pending EP PR merging]" title prefix, un-draft, paste the body between the markers. Format: PR 4500's (Summary / Implementation / Limitations / Tests with the deterministic comparison). Numbers marked TBD come from the 100-step runs `mx3_qb100_*` (in flight); fill from `matrix_scripts/qb_summarize.py`.

--- PASTE BEGIN ---

## Summary

Add quantile balancing for Kimi K3's MoE router bias.

- The bias is solved at each optimizer step from an accumulated per-expert load histogram at the balanced quantile (report sec. 2.3.3, eqs. 13-14, app. D), instead of nudged by a fixed sign step from the current step's counts. Same bias tensor, same hook point, no model change.
- The hook is one preallocated `(num_experts, num_bins)` histogram per MoE layer and a single `add_` on the router's scores; the solve inverts the accumulated CDF at the Top-(k+1) cutoff with mean-centred margins.
- The `kimi_k3_debugmodel_qb` flavor installs the solver as the model spec's `post_optimizer_build_fn`, the slot core's sign-rule hook occupies; under the flavor the solver is the only writer of `expert_bias_E`.

## Implementation

- Histograms use a fixed score range `[-1, 1]` with 512 bins and mirrored margins; the report's app. D uses an adaptive range and 1000 bins. On one frozen batch the exact-quantile solve and the histogram solve agree within 0.02 in load cv.
- Histograms are summed once over the loss mesh (dp_replicate x dp_shard x cp), stacked across layers into one collective, and not over tp: the router's scores are Replicate under TP, and a second sum would scale the histogram by the tp degree.
- Branch-free and allocation-free: selective activation checkpointing replays the router forward, and a lazily allocated buffer changes the op sequence between forward and recompute (observed; with this shape SAC on and off print identical losses). Full-AC replays count every token twice, and `cumsum / total` is invariant to a uniform factor.
- The solver resets the MoE's per-step token counter, which core's hook did.

## Limitations

- The flavor replaces core's load-balancing registration instead of composing with it: a model spec holds one `post_optimizer_build_fn`.
- The histogram range is fixed: scores outside `[-1, 1]` land in the end bins. The debug model's scores stay inside it.
- Not exercised with pipeline parallelism.

## Tests

```text
pytest tests/unit_tests/cpu/test_quantile_balance.py
```

Result: `14 passed` (histogram accumulation, CDF inversion, the solve against a brute-force reference, in-place zeroing, the SAC-identical op sequence). `tests/unit_tests/gpu/test_kimi_k3.py`: `2 passed, 1 skipped`.

The deterministic BF16 comparison used `seed=42`, `--debug.deterministic`, one seed checkpoint, 8192 tokens per rank per step in 256-token micro-batches, and 100 training steps. The sign-step reference is the debug flavor on the parent commit `65ba8a697`; quantile balancing is the qb flavor on this commit. Percentages are relative to the sign-step run of the same parallelism; dp8 x ep8 is the K3 layout (32 experts, 4 per rank, the histogram summed over the 8-rank loss mesh).

| Step | dp8 x ep8 sign-step loss | dp8 x ep8 QB loss (diff) | dp1 sign-step loss | dp1 QB loss (diff) | dp8 x ep8 sign-step grad norm | dp8 x ep8 QB grad norm (diff) | dp1 sign-step grad norm | dp1 QB grad norm (diff) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | TBD | TBD | `12.518870` | `12.518870` (`0%`) | TBD | TBD | TBD | TBD |
| 10 | TBD | TBD | `3.062460` | `2.854070` (`6.80%`) | TBD | TBD | TBD | TBD |
| 100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Step 1 is identical under both hooks by construction (the bias is 0 before the first solve). As a control, the sign-step flavor on this commit matched the parent commit's run at all 100 steps, with and without the load probe that produced the table below. Every rank holds the same solved bias after every step (all-gathered and compared, ep8 included).

What the change is for is the load. Expert load per MoE layer (tokens routed to each expert, summed over the loss mesh) as the coefficient of variation over the 32 experts and the largest and smallest expert's load relative to the mean, averaged over the 23 MoE layers and over the step window; the bias range at step 100:

| config | hook | cv steps 1-10 | cv 41-50 | cv 91-100 | max/mean 91-100 | min/mean 91-100 | bias range at 100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dp8 x ep8 | sign-step (main) | TBD | TBD | TBD | TBD | TBD | TBD |
| dp8 x ep8 | quantile balancing | TBD | TBD | TBD | TBD | TBD | TBD |
| dp1 | sign-step (main) | 1.32 | TBD | TBD | TBD | TBD | TBD |
| dp1 | quantile balancing | 0.94 | TBD | TBD | TBD | TBD | TBD |
| dp2 x ep2 | sign-step (main) | TBD | TBD | TBD | TBD | TBD | TBD |
| dp2 x ep2 | quantile balancing | TBD | TBD | TBD | TBD | TBD | TBD |

With the router frozen (`lr` 0, dp1, 15 steps, the same data through unchanging weights) the sign step moves the load cv from TBD to TBD and quantile balancing to TBD: the hook moves the loads, not the training.

Runs on RTX 5060 Ti (SM120) with the KDA capability check widened locally so Attention Gym's Triton path runs; the flavor is otherwise unchanged.

--- PASTE END ---
