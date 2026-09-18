Filed as #4780 on 2026-09-18, branch `k3_attnres_recompute` at `9f6bae06f`, two commits on upstream main `68c97b0c5`. It sits next to #4656, which the Relation section below addresses.

# PR body draft: correct the attention residual initialisation and stop retaining its FP32 copies

Branch `k3_attnres_recompute` = `9f6bae06f`, two commits on `upstream/main` `68c97b0c5`. Fork: `origin/k3_attnres_recompute`.

    1436053ea  aggregate the block residual without retaining FP32 copies
    9f6bae06f  zero initialise the attention residual projections

Context (not for pasting): deliberately separate from PR 4312. The aggregation came in with #4025 and sits on main today; its three call sites carry no pipeline guard, so both problems apply at dp1 on a single card. Filing this inside a pipeline PR that is waiting for review would move that diff and add an argument unrelated to pipelining.

Two things are kept apart on purpose. The initialisation is a correctness fix against a stated requirement and carries no measurement, because none is owed. Every number below is about the aggregation.

History of the initialisation half (not for pasting): added 2026-08-24 as `d54d327a9`, reverted the same day as `53b613d80`. The revert was on scope, not merit: a pipeline branch had no business changing how upstream's modules initialise, and the revert message says the observation should reach the maintainers as a question instead. This PR is that occasion.

--- PASTE BEGIN ---

## Summary

Two independent corrections to Kimi K3's attention residual, both in the aggregation that #4025 introduced and both unrelated to parallelism.

The pseudo-query initialisation does not match the report. Section 5 of the Kimi Linear tech report states that all pseudo-query vectors must be initialised to zero, which is what makes the depth softmax start uniform and the residual reduce to a standard residual at step 0. The three residual projections are built with `trunc_normal_` at std 0.02 instead.

The aggregation retains more than it needs. `_apply_attention_residual` keeps two FP32 `[T, N + 1, D]` intermediates alive until backward, the upcast values and the normalized keys. With activation checkpointing on, which is the default, they are rebuilt during recompute as well, so they set the backward peak.

- `torchtitan/models/kimi_k3/__init__.py`: the three residual projections initialise to zero.
- `torchtitan/models/kimi_k3/model.py`: `_AttentionResidualAggregation`, registered as a local autograd Function for SPMD type checking; `_apply_attention_residual` keeps its signature and becomes a thin call into it.
- `tests/unit_tests/cpu/test_kimi_k3_attention_residual.py`: the aggregation against a plain autograd reference, a bfloat16 precision case, the registration, and the initialisation.

## Design

The forward unbinds the block stack into per-source views, computes each source's inverse RMS and its score against the query, and forms the depth softmax from those. It saves the score weight, the norm weight, the softmax weights, the scores, the inverse RMS and references to the two inputs. None of the saved statistics carries a hidden-size factor. The backward recomputes one FP32 upcast per source and applies the softmax Jacobian, the score path and the variance path explicitly, accumulating the query gradient as a GEMV per source so the reduction order is fixed.

Both factors of the score weight are upcast before multiplying. Multiplying them in bfloat16 and upcasting the product moves the output by about half a percent at a realistic logit scale, which is larger than the reordering this change introduces.

The Function is registered with `register_local_autograd_function`. It runs no collective, leaves the token dimension alone, and reduces over the stack and hidden axes, neither of which is sharded; the two parameters are declared TP unsharded. Without the registration the type checker raises in strict mode, which the multimodal cell runs under.

At zero initialisation the scores are zero, the depth softmax is uniform, and the aggregation returns the mean of its sources. The projection still receives a gradient and moves off zero on the first step; the norm weight reaches the loss only through its product with the projection, so it gains one once the projection is nonzero.

## Compilation

Nothing here interacts with `torch.compile`, and the custom Function does not stand in its way.

`parallelize_kimi_k3` calls `torch.compile` nowhere today, so this path does not exist for Kimi K3 as the tree stands. The change touches `model.py`, the model's `__init__.py` and one test file, none of which is compile related. The Kimi K3 model carries no `compile_with_inductor` annotation either, so the regional inductor backend in `distributed/compile.py` has no region near the aggregation to lower.

If per-block compilation is turned on later it arrives as `transformer_block.compile(backend=backend, fullgraph=True)`, where a graph break is an error rather than a fallback. Checked with `torch._dynamo.explain` on the aggregation itself, the Function traces into one graph with no break, and the op count drops from 8 to 2. That is the aggregation in isolation rather than a whole block under `fullgraph=True`, which cannot be run here while the model does not compile.

## Relation to #4656

The two touch the same function from opposite sides and compose. #4656 wraps `_apply_attention_residual` in a checkpoint or, under RegionAC, declares it as a named region whose output the save policy may keep instead of replaying. This change rewrites the body of that function. All four combinations run.

They are complementary rather than alternative. Keeping the region saved costs whatever the region retains, and this change is what makes that cheap: on one aggregation at 2048 tokens and `D` 7168, a saved region costs 1036.2 MiB without this change and 28.2 MiB with it.

## Results

One aggregation, bfloat16, H100 80GB, one shape per process, 10 warmups and the median of 7. Shapes run from the debug flavor up to the released model's hidden size and block count, `dim` 7168 with a stack of 8. `kept` counts the storages autograd holds that are not already live as inputs, and the two peak columns are increments over the allocation already standing when the call begins.

`tokens` is `training.num_tokens_per_microbatch_per_dp_rank`, described in the config as the number of input-token slots processed per data-parallel rank in one model forward, before context or tensor parallel sharding. Kimi K3's hidden states are token-major `[T, D]`, so it is the leading dimension of every tensor below. For reference the released flavor sets it to one times `max_context_length`, which is 262144, so even the widest row here is an eighth of that default.

### With activation checkpointing, which is the default

The block checkpoint discards the saved tensors, so neither form retains anything and the difference is entirely in peak. The naive form still builds both FP32 tensors in forward, and builds them a second time during recompute.

| tokens | dim | stack | form | kept MiB | fwd peak MiB | bwd peak MiB | fwd ms | bwd ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 1024 | 2 | main | 0.0 | 100.0 | 144.1 | 0.52 | 1.27 |
| 2048 | 1024 | 2 | this | 0.0 | 40.1 | 68.2 | 0.63 | 1.46 |
| 2048 | 4096 | 4 | main | 0.0 | 560.1 | 960.0 | 1.00 | 2.88 |
| 2048 | 4096 | 4 | this | 0.0 | 160.2 | 256.3 | 1.16 | 2.87 |
| 2048 | 7168 | 8 | main | 0.0 | 1764.1 | 3024.0 | 2.51 | 7.99 |
| 2048 | 7168 | 8 | this | 0.0 | 280.4 | 644.6 | 2.99 | 7.92 |
| 8192 | 7168 | 8 | main | 0.0 | 7056.3 | 12096.0 | 9.26 | 30.39 |
| 8192 | 7168 | 8 | this | 0.0 | 1121.4 | 2578.1 | 10.02 | 27.52 |
| 32768 | 7168 | 8 | main | 0.0 | 28225.2 | 48384.0 | 36.50 | 120.83 |
| 32768 | 7168 | 8 | this | 0.0 | 4485.7 | 10312.0 | 37.85 | 106.06 |

Forward peak falls 6.3x and backward peak 4.7x once the stack reaches 8, and both ratios hold as the context grows. At the widest shape a single aggregation's backward peak is 47.25 GiB against 10.07 GiB, a difference of 37.2 GiB on one call.

### Without activation checkpointing

Here the retained bytes are the visible difference, and they stop growing with the shape.

| tokens | dim | stack | form | kept MiB | fwd peak MiB | bwd peak MiB | fwd ms | bwd ms | dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 1024 | 2 | main | 48.1 | 100.1 | 120.0 | 0.32 | 0.71 | |
| 2048 | 1024 | 2 | this | 0.1 | 40.1 | 68.1 | 0.44 | 0.76 | 0.195% |
| 2048 | 4096 | 4 | main | 320.1 | 560.1 | 639.9 | 0.86 | 2.10 | |
| 2048 | 4096 | 4 | this | 0.1 | 160.2 | 256.2 | 1.00 | 1.77 | 0.195% |
| 2048 | 7168 | 8 | main | 1008.2 | 1764.2 | 2015.8 | 2.37 | 5.85 | |
| 2048 | 7168 | 8 | this | 0.2 | 280.4 | 644.4 | 2.83 | 4.97 | 0.220% |
| 8192 | 7168 | 8 | main | 4032.6 | 7056.6 | 8063.4 | 9.13 | 22.39 | |
| 8192 | 7168 | 8 | this | 0.8 | 1121.4 | 2577.2 | 9.89 | 17.64 | 0.206% |
| 32768 | 7168 | 8 | main | 16130.3 | 28226.3 | 32253.7 | 36.34 | 89.06 | |
| 32768 | 7168 | 8 | this | 3.4 | 4485.7 | 10308.6 | 37.69 | 68.25 | 0.355% |

48.1, 320.1, 1008.2, 4032.6 and 16130.3 MiB become 0.1, 0.1, 0.2, 0.8 and 3.4.

Forward is consistently slower, between 3% and 21%, because a few large operations become `N + 1` smaller ones. Backward is slower at the smallest shape and faster from `dim` 4096 upward, reaching 1.14x with checkpointing on at the widest shape.

The result is not bitwise. The reordering takes the dot product first and scales by the inverse RMS where the current form normalizes and then contracts, which lands at about one bfloat16 ulp. In FP32 the two agree exactly, forward and every gradient, which the CPU test asserts at the default tolerance.

### The released model, computed rather than measured

93 layers at block size 12 gives 8 blocks and 186 aggregations, and an aggregation inside block `b` runs over `b + 2` values. Each call retains two FP32 `[T, b + 2, D]` tensors today against three FP32 `[b + 2, T]` statistics after, so the ratio is the hidden size. Computed from that structure at `dim` 7168 with pipeline degree 8, with no block level checkpointing, where retained bytes are what the change removes:

| tokens | aggregations per rank | retained today | retained after |
| --- | --- | --- | --- |
| 4096 | 23 | 27.6 GiB | 0.006 GiB |
| 8192 | 23 | 55.2 GiB | 0.012 GiB |
| 32768 | 23 | 220.7 GiB | 0.046 GiB |

Under the default checkpointing that residency is already absorbed by the block checkpoint, and what the change removes instead is the peak of a single aggregation, which does not accumulate across layers. The measured peaks in the first table are the relevant figures there.

### Where the naive form stops fitting

Same hidden size and stack, one process per point, peak reported as the process total rather than as an increment, so these numbers are larger than the two tables above and are not comparable to them.

| tokens | main | this change |
| --- | --- | --- |
| 32768 | 52.06 GiB | 14.88 GiB |
| 65536 | out of memory | 29.77 GiB |
| 131072 | out of memory | 59.53 GiB |
| 262144 | out of memory | out of memory |

On a 79.18 GiB card the naive form stops at 32768 tokens and this change reaches 131072, four times further. Neither reaches 262144, so this raises the context a single rank can carry through one aggregation and does not make the released default fit on one card; that configuration is sharded by context, tensor and pipeline parallelism, and `tokens` here is counted before any of that sharding.

## Test plan

    pytest tests/unit_tests/cpu/test_kimi_k3_attention_residual.py -q
    6 passed, 2 subtests passed

The cases are the aggregation against a plain autograd reference in FP32 at stack widths 1 and 3, a bfloat16 case that fails if the score weight is rounded before the upcast, a check that the Function is registered for SPMD type checking, and three for the initialisation: the uniform depth weights, the gradient behaviour of the two parameters, and that all three residual projections carry a zero initialiser.

The whole CPU suite was also run against a worktree at unmodified `upstream/main`, since a count from any other tree is not a baseline. Failures and errors are identical on both sides, 19 and 7, the same files in both cases and none of them a Kimi K3 test; the seven errors are missing packages in that environment. Passed differs only by the tests this PR adds, and the new file was confirmed present in the collection list rather than inferred from the counts.

`tests/unit_tests/gpu/test_kimi_k3.py` has no case covering the aggregation, so the GPU suite does not exercise this path today.

## Limitations

The initialisation change alters training behaviour from step zero for a model initialised from scratch. It does not affect a run that loads a checkpoint.

`@once_differentiable` on the backward removes double backward, which the plain autograd form supported. Nothing in the tree uses `create_graph`, and `models/common/linear.py`, `overrides/fused_mla.py` and the MXFP8 linear already use the decorator.

A step 1 gradient comparison at model level is not included. The Attention Gym KDA kernel accepts only CUDA capability 10.0 and 10.3, so a full Kimi K3 step runs on neither an H100 nor an SM120 card without relaxing that guard, and a number produced under a relaxed guard is not reproducible from an unmodified tree. The operator level comparison stands in its place: exact in FP32 and one ulp in bfloat16.

--- PASTE END ---
