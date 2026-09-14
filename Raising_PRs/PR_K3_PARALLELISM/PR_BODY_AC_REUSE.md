# PR title: [Kimi K3] Recompute the attention-residual math in backward

Fork branch `k3_ac_reuse_attention` = `a3e7d857c` (three commits on main `b21f7d43e`: the recompute, the region declarations, the tests). Measured on 8 x RTX 5060 Ti with the KDA capability guard lifted locally for the run; the lift is not part of the branch.

--- PASTE BEGIN ---

## Summary

Recompute Kimi K3's attention-residual math in backward, as described in section 5.2.2 of the Kimi K3 technical report ("The AttnRes computation is entirely wrapped with checkpointing, so the activation saved for the backward pass at each layer is identical to that of the standard residual architecture").
- Run `_apply_attention_residual` under a `torch_remat` checkpoint in `kimi_k3/model.py` whenever autograd records it, for the two residuals of every block and the output aggregation.
- Declare `torch_remat` regions for the MLA and KDA call sites and the two block residuals, so RegionAC can keep the attention activations and recompute the MoE or feed-forward by name.
- Add CPU tests for the recompute (bitwise gradients, no stack-shaped saved tensor) and for the regions under RegionAC.

## Design

The residual upcasts the whole block stack and the prefix sum to fp32 and keeps the (N+1)-entry intermediates for backward, so each layer's saved activations grow with the stack. Under a checkpoint, backward recomputes them from the stack and the prefix sum, which are kept anyway, so the per-layer saved set matches a standard residual block at the cost of re-running the residual math. Values are unchanged.

The recompute needs no activation-checkpointing mode: with AC off and under selective or full AC it is a `torch_remat` checkpoint (it nests inside the torch checkpoint those modes use). Under RegionAC the block is already a `torch_remat` checkpoint, which cannot nest another one, so a block that RegionAC configures (`configure_remat_regions`, which RegionAC calls on every block) runs its residuals as the regions `attention_res` and `ffn_res` instead, recomputed unless a save policy keeps them.

The MLA regions (`attention.q`, `attention.kv`, `attention.inner_attention`, `attention.gate`, `attention.wo`) and the KDA regions (`delta_attention.forget`, `.beta`, `.qkv`, `.inner_kda`, `.output_gate`, `.output_norm`, `.output_proj`) are an optional policy on top: `activation-checkpoint:region --activation-checkpoint.save-regions '*attention.*'` keeps the attention activations, so the KDA and MLA kernels are not re-run in backward, and recomputes the rest of the block. The pattern matches neither residual region nor the vision tower's regions (`attn.*`, `mlp.*`). Everything lives in the model folder: the residual is Kimi K3's own computation, and the regions follow the call-site pattern of the core attention and feed-forward modules.

## Results

Main `b21f7d43e` against this PR, `seed=42`, deterministic, one inductor cache shared by every cell and warmed by a 1-step run of each. Debug model at dp1: 2048 tokens per step in 512-token micro-batches, 10 steps. The multimodal FSDP 2 recipe of the b200 suite (`kimi_k3_debugmodel_mm_fsdp2`, SPMD type checking on) with activation checkpointing forced off: its default 2048 tokens per rank, 3 steps. Loss and grad norm are compared at every step.

```bash
PYTHONPATH=<attention-gym main>:. torchrun --nproc_per_node=1 -m torchtitan.train \
  --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
  --training.steps 10 --training.num-tokens-per-train-step 2048 \
  --training.num-tokens-per-microbatch-per-dp-rank 512 activation-checkpoint:none
```

| cell | activation checkpointing | main | this PR | loss and grad norm | peak memory main / PR | tps main / PR |
| --- | --- | --- | --- | --- | ---: | ---: |
| debug dp1 | none | `12.63048` / `20.6250` at step 1, `3.64293` / `4.9062` at step 10 | same | bitwise | 14.61 / 14.17 GiB | 853 / 774 |
| debug dp1 | selective (the flavor default) | same | same | bitwise | 12.68 / 12.72 GiB | 504 / 465 |
| debug dp1 | region, `save-regions '*attention.*'` | | same | bitwise | 13.18 GiB | 640 |
| debug dp1 | full | | same | bitwise | 12.52 GiB | 604 |
| mm FSDP 2, type checking on | none | `12.37844` / `24.5000` at step 1, `9.62056` / `18.0000` at step 3 | same | bitwise | 15.04 / 14.76 GiB | not measured (3 steps) |

With activation checkpointing off the recompute saves 0.44 GiB of peak memory at 512-token micro-batches and 0.28 GiB on the FSDP 2 recipe at 2048 tokens, and costs about 9% of throughput at dp1 for re-running the residual math. Under selective AC the residual is recomputed inside the block's own recompute, so peak memory is unchanged within 0.04 GiB and the step is about 8% slower. The FSDP 2 recipe with type checking on also confirms the `torch_remat` checkpoint runs under SPMD type checking. The `*attention.*` row is the optional region policy: it keeps the MLA and KDA activations and runs faster than selective AC for 0.46 GiB more.


## Test plan
- `pytest tests/unit_tests/cpu/test_kimi_k3_remat_regions.py -q` (`5 passed, 5 subtests passed`)
- Scoped pre-commit checks, including formatting and Pyrefly (`passed`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
