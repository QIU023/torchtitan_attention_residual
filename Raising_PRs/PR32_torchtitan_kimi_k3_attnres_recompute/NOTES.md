# Filing notes: attention residual recompute (PR 32)

Branch `k3_attnres_recompute` = `1436053ea`, one commit on `upstream/main` `68c97b0c5`. Fork remote `origin`. Not filed yet.

## Why this is not part of PR 4312

The aggregation arrived with #4025 and sits on main today. Its three call sites carry no pipeline guard and the model file imports nothing from the pipeline, the stage or the layout, so the two retained FP32 tensors exist at dp1 on a single card, under TP and under EP. 4312's own change to `model.py` is +40/-23 and is a contract adaptation: `prefix_sum_TD` becomes an optional `partial_block_TD`, the concatenation becomes a ternary, the block boundary concatenation moves, and call site variables are renamed. The softmax, the upcast, the inverse RMS and the matmul are untouched. Filing this inside 4312 would move a review diff that is waiting on a reviewer and would add an argument unrelated to pipelining.

## Diff audit

Run against `git diff upstream/main..HEAD` before calling the draft ready.

- No logbook path, no cross repository reference form and no `.md` link in any source file.
- No measured value, benchmark result or experiment record in any docstring. An earlier grep appeared to find some; those were substring matches on `Platforms` and `inverse_rms`, not numbers.
- Added lines in `torchtitan/`: 94 code, 1 comment, 4 docstring, so docstrings are 4.0% of the change. #4577, the house reference, is 2.4%. The class docstring was cut from six lines to four after the first count read 5.9%.
- One new private definition, `_AttentionResidualAggregation`, with its forward and backward.

## Reuse check

Searched `models/common`, `components`, `distributed`, `quantization` and the sibling models for an existing depth softmax aggregation or a reusable recomputing wrapper. There is none. What core does have is the pattern this follows: `torch.autograd.Function` appears in `components/loss.py`, `models/common/linear.py`, `models/common/dist_gemm.py`, `models/common/aux_loss.py`, `models/gpt_oss/moe.py` and the quantization modules, and `@once_differentiable` is already used by `models/common/linear.py`, `overrides/fused_mla.py` and the MXFP8 linear. The SPMD registration follows `models/common/linear.py`, which is the one of those that registers.

## Evidence

Results in the body are the H100 sweep, one shape per process, 10 warmups and the median of 7, from the debug flavor's shape up to the released model's hidden size and block count at four context lengths. The 5060 was used for development and agrees in direction; none of its numbers are quoted, per the rule that body numbers come from the H100.

## Open

- A model level step 1 gradient comparison. The Attention Gym KDA kernel accepts only CUDA capability 10.0 and 10.3, so a full Kimi K3 step runs on neither the H100 (SM90) nor the 5060 Ti (SM120) without relaxing that guard, and a number produced under a relaxed guard is not reproducible from an unmodified tree. The body says so rather than omitting it.
- The full CPU suite on this branch, for the test plan.
