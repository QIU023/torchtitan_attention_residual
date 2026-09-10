# PR title: [Kimi K3] Multi-token prediction layers

Fork branch `k3_mtp_layers` = `937fe9276` (the fork already has an old-tree `k3_mtp` at `5ce30dbe1`, left untouched) (one commit on main `ac10ca48f`, independent of the parallelism PRs; MTP under pipeline parallelism is the integration tree's follow-up, not here). Verified on this box with the CPU tests and the dp1 smoke below.

--- PASTE BEGIN ---

## Summary

Kimi K3's report (sec 3.3) trains with multi-token prediction: an MTP layer mirrors one backbone block, and depth k predicts the token k+1 ahead from the backbone's final hidden state fused with the embedding of the token k+1 ahead. The released config ships zero MTP layers, so this is opt-in: `model_registry(flavor, num_mtp_layers=N)` and the `kimi_k3_debugmodel_mtp` recipe (one layer).

- `mtp.py`: `KimiK3MTPLayer` (`enorm` / `hnorm` on the two inputs, `eh_proj` from `2 x dim` back to `dim`, then one block with a backbone layer's structure) and `KimiMTPLoss`, the main next-token cross entropy plus `mtp_weight` (0.3) times the mean per-depth cross entropy; the loss reduces to exactly its inner loss when no depth logits arrive, so one loss config serves flavors with and without MTP.
- The mirrored block is KDA-typed with `layer_id=0`: it opens its own empty block stack rather than joining the backbone's attention-residual depth mixing, and KDA consumes the depth-shortened sequence directly (an MLA mirror would need a FlexAttention mask rebuilt per depth, left as the known TODO rather than a silent fallback). Embedding and head are shared with the backbone.
- The depth-k input fuses the backbone's final PRE-norm hidden state with the embedding of the token k+1 ahead (the reference feeds `hnorm` the unnormalised state; normalising twice is not an identity). The last k+1 positions have no target and are dropped, not padded. On a packed stream the loss masks depth targets that cross a document boundary using the position ids.
- The MTP layers register after the vision encoder, so a model without MTP draws the same init stream and its step-1 loss is unchanged; with `--loss.mtp_weight 0.0` the MTP flavor reproduces the plain model's loss.
- Guards: the model raises on MTP with chunked loss (MTP needs full-vocab logits per depth, exactly the allocation chunked loss exists to avoid; skipping silently would look like MTP trains while it does not), on a pipeline split that separates the embedding from the head, and on sequence or context parallelism (the depth-shifted slice indexes tokens on a stream those shard by position).

## Implementation

`__init__.py` builds the mirror block config from the same free parameters as the backbone (`_kimi_k3_config(num_mtp_layers=...)`; the mirror's `ffn_res_proj` is zero-initialised). `model.py` holds the layers in a `ModuleDict`, counts the mirror KDA's flops, computes the per-depth logits after the backbone's last block and hands them to the loss through a rank-local hand-off (`put_mtp_logits` / `take_mtp_logits`), since the trainer's loss signature carries only the main logits. `sharding.py` gives the mirror block's KDA the same local boundary as the backbone layers'. The recipe uses the plain (non-chunked) cross entropy.

## Limitations

- No HF mapping for the MTP weights (the released config ships none); `to_hf` covers the backbone only.
- Not composed with sequence parallelism, context parallelism, or a pipeline split between the embedding and the head; these raise.
- The MLA-typed mirror is not implemented (KDA-typed only).

## Tests

```text
pytest -q tests/unit_tests/cpu/test_kimi_k3_mtp_layer.py tests/unit_tests/cpu/test_integration_test_definitions.py tests/unit_tests/cpu/test_optimizer_param_groups.py
```

Result: 37 passed (the two MTP build tests: the spec with `num_mtp_layers=1` builds one KDA-typed, dense-FFN mirror layer with `eh_proj` of shape `(dim, 2 x dim)`; the default spec has no MTP layers). pre-commit passes on the touched files; `pyrefly check` (0.45.1, the repo's pin) reports the same 21 errors as main `ac10ca48f`, none in the touched files.

## Results

dp1 on one GPU (8 x RTX 5060 Ti, one used; 16 GB), the debug recipe with 1024 tokens per step (four 256-token micro-batches), bf16, `seed=42`, deterministic, 3 steps; `kimi_k3_debugmodel` (chunked cross entropy) against `kimi_k3_debugmodel_mtp` (one MTP layer, plain cross entropy, `mtp_weight` 0.3, and the same flavor with `--loss.mtp_weight 0.0`).

| cell | step 1 loss / grad norm | step 2 | step 3 | peak memory |
| --- | ---: | ---: | ---: | ---: |
| `kimi_k3_debugmodel` | `12.51200` / `19.6250` | `10.71575` / `16.7500` | `8.24823` / `11.5000` | 12.63 GiB |
| `kimi_k3_debugmodel_mtp`, `mtp_weight` 0.3 | `16.24698` / `20.2500` | `13.97159` / `16.7500` | `10.58028` / `18.7500` | 14.04 GiB |
| `kimi_k3_debugmodel_mtp`, `mtp_weight` 0.0 | `12.51200` / `19.6250` | `10.67570` / `16.1250` | `8.25151` / `10.9375` | 14.04 GiB |

The MTP flavor's loss is the composite `main CE + 0.3 x depth-1 CE`; at step 1 the depth term is `(16.24698 - 12.51200) / 0.3 = 12.45`, a freshly initialised depth head at chance over the 163840-token vocabulary (`ln 163840 = 12.0`). With the weight at 0 the flavor reports the plain model's step-1 loss and grad norm exactly, which is the init-stream claim: the MTP layer leaves the backbone's init and forward untouched. From step 2 the two cells differ by 0.4% and 0.04% in loss; besides the zero-weighted MTP branch they differ in the loss implementation (chunked cross entropy in the plain recipe, non-chunked in the MTP recipe). The MTP layer costs 1.4 GiB here: the mirror block plus the full-vocab logits of the main head and the depth, which the chunked loss otherwise avoids.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
