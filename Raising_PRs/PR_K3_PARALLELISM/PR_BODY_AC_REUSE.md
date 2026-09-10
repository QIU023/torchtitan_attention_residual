# PR title: [Kimi K3] Activation checkpointing that reuses the attention activations, and a recompute wrapper for the attention residual

Fork branch `k3_ac_reuse_attention` = `6ef880995` (three commits on main `ac10ca48f`, independent of the parallelism PRs). Verified on this box (8 x RTX 5060 Ti) with the tests and the dp1 smoke below; the KDA capability guard was lifted locally for the run and is not part of the branch.

--- PASTE BEGIN ---

## Summary

Two memory-side changes to Kimi K3's activation checkpointing, both bitwise against the current code (checkpointing recomputes, it does not change the math):

- `ac_reuse_attention` (a model config flag, off by default): under selective AC, checkpoint only each block's MoE / feed-forward and keep attention and the residual math outside the wrap, so their activations are saved once and reused in backward. The KDA kernel is a custom op outside the per-op policy's save set, so a whole-block wrap recomputes it in backward; wrapping just the FFN keeps the mm save/recompute balance where the parameter memory is and stops re-running the attention kernels. It trades activation memory for that.
- The attention-residual computation runs under `torch.utils.checkpoint` (always on, no flag): the residual math upcasts the whole (N+1)-entry block stack to fp32 twice per layer, and saving those intermediates makes each layer's activation footprint grow with the stack. Recomputing them in backward from the stack and the prefix sum, both alive elsewhere, makes the activations saved per layer the same as a standard residual architecture.

## Implementation

- `torchtitan/models/kimi_k3/model.py`: `KimiK3Model.Config.ac_reuse_attention: bool = False`; `_apply_attention_residual` becomes the checkpoint wrapper around `_attention_residual_math` (the previous body, unchanged), taken only when grad is enabled and an input requires grad, so inference and the no-grad paths call the math directly.
- `torchtitan/models/kimi_k3/parallelize.py`: with the flag, `_apply_ac_outside_attention` wraps `layers.<i>.moe` or `layers.<i>.feed_forward` with the configured policy's `_wrap_block` (the same `base_fqn` scheme the policy uses for whole blocks) instead of `ac_policy.apply(model)`; the vision tower keeps its own `apply`. The config and the layer dict are narrowed with `isinstance` for the checker.

## Limitations

- `ac_reuse_attention` is a model config field, not a CLI flag: a recipe sets it on `model_spec.model`. No recipe in the tree turns it on; the smoke below used a local alias.
- Per-layer AC (`ac_freq`) and full AC are unchanged; the flag is read only on the selective path where the whole-block wrap is what recomputes the kernels.
- The residual wrapper uses `use_reentrant=False`; the recompute runs the fp32 upcasts a second time in backward (the trade the wrapper makes).

## Tests

```text
pytest -q tests/unit_tests/cpu/test_kimi_k3_attn_res_checkpoint.py tests/unit_tests/cpu/test_integration_test_definitions.py
```

Result: 16 passed (3 new: the wrapped residual's values and gradients equal the unwrapped math at `rtol=0, atol=0`; the body runs a second time in backward rather than being read back; under `no_grad` the wrapper is skipped and the values still match). pre-commit passes on the touched files; `pyrefly check` (the pinned 0.45.1) reports the same 21 errors as main `ac10ca48f`, none in the touched files.

## Results

Kimi K3 debug model, dp1, bf16, `seed=42`, deterministic, one seed checkpoint for both cells, 8192 tokens per step in 256-token micro-batches, 10 steps; the flavor as it is (selective AC over the whole block, the residual under the new wrapper) against the same flavor with `ac_reuse_attention` on.

| cell | step 1 loss / grad norm | step 3 | step 10 | peak memory | tps at step 10 |
| --- | --- | --- | --- | ---: | ---: |
| main `ac10ca48f`, selective AC over the whole block | `12.51887` / `14.1250` | `7.11252` / `10.0625` | `3.11301` / `2.0312` | 12.65 GiB | 256 |
| this branch, the same flavor (residual under the wrapper) | bitwise | bitwise | bitwise | 12.65 GiB | 246 |
| this branch, `ac_reuse_attention` on | bitwise | bitwise | bitwise | 12.78 GiB | 305 |

Both changes are bitwise against main over the ten steps. On this debug model (24 layers, 256-token micro-batches) the residual wrapper's saved-activation reduction is below the 0.01 GiB resolution of the peak reading; `ac_reuse_attention` costs 0.13 GiB of activations and takes the step from 256 to 305 tokens per second by not re-running the KDA and MLA kernels in backward (the log confirms the wrap covered the MoE / feed-forward of all 24 blocks).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
