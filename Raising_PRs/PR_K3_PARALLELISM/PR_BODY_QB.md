# PR title: [Kimi K3] Quantile balancing for the MoE router bias

PR 4412, head `k3_qb` = `qb_review4` = `895f4d6e9` (one commit squashed onto upstream/main `65ba8a697`, pushed 2026-09-09); `git diff upstream/main` is exactly `torchtitan/models/kimi_k3/quantile_balance.py` (+299), `tests/unit_tests/cpu/test_kimi_k3_quantile_balance.py` (+273) and `torchtitan/models/kimi_k3/config_registry.py` (+4). Verified on the GPU box: 14 CPU tests pass, `tests/unit_tests/gpu/test_kimi_k3.py` passes; un-draft and paste the body between the markers. Format: PR 4500's (Summary / Implementation / Limitations / Tests with the deterministic comparison). 10-step numbers from `mx3_qb10_*` (2026-09-09), complete. The 100-step batch runs overnight and replaces the tables (`matrix_scripts/qb_probe/qb_summarize.py`).

--- PASTE BEGIN ---

## Summary

Add quantile balancing for Kimi K3's MoE router bias.

- The bias is solved at each optimizer step from an accumulated per-expert load histogram at the balanced quantile (report sec. 2.3.3, eqs. 13-14, app. D), instead of nudged by a fixed sign step from the current step's counts. Same bias tensor, same hook point, no model change.
- The hook is one preallocated `(num_experts, num_bins)` histogram per MoE layer and a single `add_` on the router's scores; the solve inverts the accumulated CDF at the Top-(k+1) cutoff with mean-centred margins.
- `kimi_k3_debugmodel` registers the solver as the model spec's `post_optimizer_build_fn`, the slot core's sign-rule hook occupies, so the solve is the only writer of `expert_bias_E`. The rule is K3's, so it lives in the model folder rather than in `components`, where core's model-agnostic hook is.

## Implementation

- Histograms use a fixed score range `[-1, 1]` with 512 bins and mirrored margins; the report's app. D uses an adaptive range and 1000 bins. On one frozen batch the exact-quantile solve and the histogram solve agree within 0.02 in load cv.
- Histograms are summed once over the loss mesh (dp_replicate x dp_shard x cp), stacked across layers into one collective, and not over tp: the router's scores are Replicate under TP, and a second sum would scale the histogram by the tp degree.
- Branch-free and allocation-free: selective activation checkpointing replays the router forward, and a lazily allocated buffer changes the op sequence between forward and recompute (observed; with this shape SAC on and off print identical losses). Full-AC replays count every token twice, and `cumsum / total` is invariant to a uniform factor.
- The solver resets the MoE's per-step token counter, which core's hook did.

## Limitations

- Registering the solver replaces core's load-balancing hook instead of composing with it: a model spec holds one `post_optimizer_build_fn`.
- The histogram range is fixed: scores outside `[-1, 1]` land in the end bins. The debug model's scores stay inside it.
- Not exercised with pipeline parallelism.

## Tests

```text
pytest tests/unit_tests/cpu/test_kimi_k3_quantile_balance.py
```

Result: `14 passed` (histogram accumulation, CDF inversion, the solve against a brute-force reference, in-place zeroing, the SAC-identical op sequence). `tests/unit_tests/gpu/test_kimi_k3.py`: `2 passed, 1 skipped`.

The deterministic BF16 comparison used `seed=42`, `--debug.deterministic`, one seed checkpoint, 8192 tokens per rank per step in 256-token micro-batches, and 10 training steps. The sign-step reference is the debug config on the parent commit `65ba8a697`; quantile balancing is the same config on this commit, where it registers the solver. Percentages are relative to the sign-step run of the same parallelism; dp8 x ep8 is the K3 layout (32 experts, 4 per rank, the histogram summed over the 8-rank loss mesh).

Step 1 is identical under both hooks by construction (the bias is 0 before the first solve); from step 2 the two hooks route the same tokens to different experts, so every row below step 1 is two different runs of the same model rather than a numerics comparison, and the percentages size that divergence rather than an error.

| config | hook | step 1 | step 3 | step 10 |
| --- | --- | ---: | ---: | ---: |
| dp8 x ep8 | sign-step (main) | `12.522390` | `6.618530` | `2.754520` |
| dp8 x ep8 | quantile balancing | `12.522390` (`0%`) | `6.791680` (`2.62%`) | `2.743960` (`0.38%`) |
| dp2 x ep2 | sign-step (main) | `12.523720` | `7.502180` | `3.211300` |
| dp2 x ep2 | quantile balancing | `12.523720` (`0%`) | `7.284180` (`2.91%`) | `3.156020` (`1.72%`) |
| dp1 | sign-step (main) | `12.518870` | `7.112520` | `3.113010` |
| dp1 | quantile balancing | `12.518870` (`0%`) | `7.353260` (`3.38%`) | `3.047750` (`2.10%`) |

Gradient norms on the two configurations that carry them:

| config | hook | step 1 | step 3 | step 10 |
| --- | --- | ---: | ---: | ---: |
| dp8 x ep8 | sign-step (main) | `13.3125` | `8.6875` | `1.1953` |
| dp8 x ep8 | quantile balancing | `13.3125` (`0%`) | `10.875` (`25.2%`) | `1.3125` (`9.81%`) |
| dp1 | sign-step (main) | `14.125` | `10.0625` | `2.0312` |
| dp1 | quantile balancing | `14.125` (`0%`) | `9.5` (`5.59%`) | `2.1406` (`5.39%`) |

As a control, the sign-step hook on this commit matched the parent commit's run at every step (loss and grad norm), with and without the load probe that produced the table below. Every rank holds the same solved bias after every step (all-gathered and compared, ep8 included).

What the change is for is the load. Expert load per MoE layer (tokens routed to each expert, summed over the loss mesh) as the coefficient of variation over the 32 experts and the largest and smallest expert's load relative to the mean, averaged over the 23 MoE layers and over the step window; the bias range at step 10:

| config | hook | cv steps 1-5 | cv 6-10 | max/mean 6-10 | min/mean 6-10 | bias range at 10 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dp8 x ep8 | sign-step (main) | 1.25 | 1.49 | 5.5 | 0.01 | [-0.016, 0.006] |
| dp8 x ep8 | quantile balancing | 1.09 | 0.82 | 3.4 | 0.05 | [-0.297, 0.333] |
| dp2 x ep2 | sign-step (main) | 1.26 | 1.51 | 5.7 | 0.01 | [-0.016, 0.006] |
| dp2 x ep2 | quantile balancing | 1.13 | 0.90 | 3.6 | 0.03 | [-0.352, 0.339] |
| dp2 | sign-step (main) | 1.25 | 1.54 | 5.9 | 0.02 | [-0.016, 0.006] |
| dp2 | quantile balancing | 1.10 | 0.93 | 3.7 | 0.03 | [-0.400, 0.341] |
| dp1 | sign-step (main) | 1.20 | 1.36 | 5.1 | 0.02 | [-0.016, 0.006] |
| dp1 | quantile balancing | 1.01 | 0.80 | 3.2 | 0.05 | [-0.291, 0.344] |

With the router frozen (`lr` 0, dp1, 10 steps, the same data through unchanging weights) the sign step moves the load cv from 1.05 (steps 1-5) to 0.89 (steps 6-10) and quantile balancing from 0.76 to 0.58: the hook moves the loads, not the training.

Runs on RTX 5060 Ti (SM120) with the KDA capability check widened locally so Attention Gym's Triton path runs; nothing else in the config differs.

--- PASTE END ---
