# PR title: [Kimi K3] Recompute the attention-residual math in backward

Fork branch `k3_ac_reuse_attention` = `362b6cc70` (three commits on main `b21f7d43e`: the recompute, the region declarations, the tests). Measured on 8 x RTX 5060 Ti with the KDA capability guard lifted locally for the run; the lift is not part of the branch.

--- PASTE BEGIN ---

## Summary

Recompute Kimi K3's attention-residual math in backward, as described in section 5.2.2 of the Kimi K3 technical report ("The AttnRes computation is entirely wrapped with checkpointing, so the activation saved for the backward pass at each layer is identical to that of the standard residual architecture").
- Run `_apply_attention_residual` under a `torch_remat` checkpoint in `kimi_k3/model.py` whenever autograd records it and no activation-checkpointing policy already covers the block, for the two residuals of every block, and always for the output aggregation.
- Mark the blocks in `parallelize_kimi_k3` when selective, full or region AC checkpoints them, so their residuals run as plain calls inside that checkpoint.
- Declare `torch_remat` regions for the MLA and KDA call sites and the two block residuals, so RegionAC can keep the attention activations and recompute the MoE or feed-forward by name.
- Add CPU tests for the recompute (bitwise gradients, no stack-shaped saved tensor) and for the regions under RegionAC.

## Design

The residual upcasts the whole block stack and the prefix sum to fp32 and keeps the (N+1)-entry intermediates for backward, so each layer's saved activations grow with the stack. Under a checkpoint, backward recomputes them from the stack and the prefix sum, which are kept anyway, so the per-layer saved set matches a standard residual block at the cost of re-running the residual math. Values are unchanged.

The saved set matches a standard residual block in every mode. With activation checkpointing off, the model guarantees it: each residual is a `torch_remat` checkpoint. Selective and full AC wrap the whole block in a torch checkpoint, which already keeps these intermediates out of the saved set, so `parallelize_kimi_k3` marks the block (`checkpoint_residual = False`) and its residuals run as plain calls; a second checkpoint inside would only recompute them again. Under RegionAC the block is a `torch_remat` checkpoint, which cannot nest another one, so a block that RegionAC configures (`configure_remat_regions`) runs its residuals as the regions `attention_res` and `ffn_res`, recomputed unless a save policy keeps them. The output aggregation on the head stage sits outside every block checkpoint and always takes its own.

The MLA regions (`attention.q`, `attention.kv`, `attention.inner_attention`, `attention.gate`, `attention.wo`) and the KDA regions (`delta_attention.forget`, `.beta`, `.qkv`, `.inner_kda`, `.output_gate`, `.output_norm`, `.output_proj`) are an optional policy on top: `activation-checkpoint:region --activation-checkpoint.save-regions '*attention.*'` keeps the attention activations, so the KDA and MLA kernels are not re-run in backward, and recomputes the rest of the block. The pattern matches neither residual region nor the vision tower's regions (`attn.*`, `mlp.*`). Everything lives in the model folder: the residual is Kimi K3's own computation, and the regions follow the call-site pattern of the core attention and feed-forward modules.

## Results

Main `b21f7d43e` against this PR on the debug model, dp1, bf16, `seed=42`, deterministic, 2048 tokens per step in 512-token micro-batches, 10 steps, one GPU, one inductor cache shared by every cell and warmed by a 1-step run of each. Loss and grad norm are compared at all ten steps.

```bash
PYTHONPATH=<attention-gym main>:. torchrun --nproc_per_node=1 -m torchtitan.train \
  --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
  --training.steps 10 --training.num-tokens-per-train-step 2048 \
  --training.num-tokens-per-microbatch-per-dp-rank 512 activation-checkpoint:none
```

| activation checkpointing | loss and grad norm, main vs PR | peak memory main / PR | tps main / PR |
| --- | --- | ---: | ---: |
| none | bitwise (`12.63048` / `20.6250` at step 1, `3.64293` / `4.9062` at step 10) | 14.61 / 14.17 GiB | 838 / 788 |
| selective (the flavor default) | bitwise | 12.68 / 12.72 GiB | 501 / 501 |
| full | bitwise | 12.52 / 12.52 GiB | 608 / 616 |
| region, `save-regions '*attention.*'` | bitwise | 12.52 / 13.18 GiB | 639 / 658 |

With activation checkpointing off the recompute saves 0.44 GiB of peak memory for about 6% of throughput. Selective and full AC run as on main, because their block checkpoint already excludes the intermediates and the residual is not checkpointed a second time. On main the `*attention.*` pattern matches no Kimi K3 region, so RegionAC recomputes the whole block; with the regions of this PR it keeps the MLA and KDA activations for 0.66 GiB and is the fastest checkpointed mode.

## Test plan
- `pytest tests/unit_tests/cpu/test_kimi_k3_remat_regions.py -q` (`6 passed, 5 subtests passed`)
- Scoped pre-commit checks, including formatting and Pyrefly (`passed`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
