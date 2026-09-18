# PR body draft: recompute the Kimi K3 attention residual in backward

Branch `k3_attnres_recompute` = `cf4067f19`, one commit on `upstream/main` `68c97b0c5`. Fork: `origin/k3_attnres_recompute`.

Context (not for pasting): this is deliberately separate from PR 4312. The aggregation came in with #4025 and sits on main today; its three call sites carry no pipeline guard, so the two retained FP32 tensors exist at dp1 on a single card. 4312's change to `model.py` is a contract adaptation that leaves the arithmetic untouched. Filing this inside a pipeline PR that is waiting for review would move that diff and add an argument unrelated to pipelining. Results below are from the H100 box; the 5060 numbers agree in direction and are not quoted.

--- PASTE BEGIN ---

## Summary

`_apply_attention_residual` keeps two FP32 `[T, N + 1, D]` intermediates alive until backward, the upcast values and the normalized keys, which at the released model's shape is 1008 MiB for a single call and 15.75 GiB at a long context. This replaces the body with an `autograd.Function` that saves only per-token statistics and recomputes the rest in backward, leaving the retained bytes independent of the hidden size.

- `torchtitan/models/kimi_k3/model.py`: `_AttentionResidualAggregation`, registered as a local autograd Function for SPMD type checking; `_apply_attention_residual` keeps its signature and becomes a thin call into it.
- `tests/unit_tests/cpu/test_kimi_k3_attention_residual.py`: the aggregation against a plain autograd reference, a bfloat16 precision case, and a registration check.

## Design

The forward unbinds the block stack into per-source views, computes each source's inverse RMS and its score against the query, and forms the depth softmax from those. It saves the score weight, the norm weight, the softmax weights, the scores, the inverse RMS and references to the two inputs. None of the saved statistics carries a hidden-size factor, and the two inputs are already live, so the call retains essentially nothing. The backward recomputes one FP32 upcast per source and applies the softmax Jacobian, the score path and the variance path explicitly, accumulating the query gradient as a GEMV per source so the reduction order is fixed.

The score weight is the product of the norm weight and the projection weight. Both factors are upcast before multiplying. Multiplying them in bfloat16 and upcasting the product moves the output by about half a percent at a realistic logit scale, which is larger than the reordering this change introduces.

The Function is registered with `register_local_autograd_function`. It runs no collective, leaves the token dimension alone, and reduces over the stack and hidden axes, neither of which is sharded; the two parameters are declared TP unsharded. Without the registration the type checker raises in strict mode, which the multimodal cell runs under.

## Relation to #4656

The two touch the same function from opposite sides and compose. #4656 wraps `_apply_attention_residual` in a checkpoint or, under RegionAC, declares it as a named region whose output the save policy may keep instead of replaying. This change rewrites the body of that function. All four combinations run.

They are complementary rather than alternative. Keeping the region saved costs whatever the region retains, and this change is what makes that cheap: measured on one aggregation at 2048 tokens and `D` 7168, a saved region costs 1036.2 MiB without this change and 28.2 MiB with it.

## Results

One aggregation, bfloat16, H100 80GB, one shape per process, 10 warmups and the median of 7. `kept` counts the storages autograd holds that are not already live as inputs. `dev` is the maximum absolute difference from the current form relative to its own maximum. The shapes run from the debug flavor up to the released model's hidden size and block count, `dim` 7168 with a stack of 8, at four context lengths.

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

Retained bytes stop growing with the shape: 48.1, 320.1, 1008.2, 4032.6 and 16130.3 MiB become 0.1, 0.1, 0.2, 0.8 and 3.4. Forward peak falls 6.3x and backward peak 3.1x once the stack reaches 8, since the FP32 tensors are never built rather than built and discarded.

Forward is consistently slower, between 3% and 16%, because a few large operations become `N + 1` smaller ones. Backward is faster everywhere except the smallest shape: 1.19x, 1.18x, 1.27x and 1.30x against 0.93x at `dim` 1024 with a stack of 2.

The result is not bitwise. The reordering takes the dot product first and scales by the inverse RMS where the current form normalizes and then contracts, which lands at about one bfloat16 ulp. In FP32 the two agree exactly, forward and every gradient, which the CPU test asserts at the default tolerance.

## Test plan

    pytest tests/unit_tests/cpu/test_kimi_k3_attention_residual.py -q
    3 passed, 2 subtests passed

The three cases are the aggregation against a plain autograd reference in FP32 at stack widths 1 and 3, a bfloat16 case that fails if the score weight is rounded before the upcast, and a check that the Function is registered for SPMD type checking.

`tests/unit_tests/gpu/test_kimi_k3.py` has no case covering the aggregation, so the GPU suite does not exercise this path today.

## Limitations

`@once_differentiable` on the backward removes double backward, which the plain autograd form supported. Nothing in the tree uses `create_graph`, and `models/common/linear.py`, `overrides/fused_mla.py` and the MXFP8 linear already use the decorator.

A step 1 gradient comparison at model level is not included. The Attention Gym KDA kernel the model uses accepts only CUDA capability 10.0 and 10.3, so a full Kimi K3 step does not run on either an H100 or an SM120 card without relaxing that guard, and a number produced under a relaxed guard is not reproducible from an unmodified tree. The operator level comparison above stands in its place: exact in FP32 and one ulp in bfloat16.

--- PASTE END ---
