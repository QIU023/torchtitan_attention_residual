# PR title: [Kimi K3] Declare torch_remat regions so RegionAC can keep attention and recompute the MoE

Fork branch `k3_ac_reuse_attention` = [`bc696be15`](https://github.com/QIU023/torchtitan/commit/bc696be15eca3fc26559c9aab76f7adde8c993a1) (two commits on main `b21f7d43e`, independent of the parallelism PRs). Replaces the earlier draft of this branch (a model flag plus a private wrap of the MoE / feed-forward under selective AC, and a direct `torch.utils.checkpoint` around the attention residual) with region declarations on main's RegionAC. Measured on `1c7ab8089`; the one newer main commit, `b21f7d43e` (#4611), only adds vision-encoder regions, which the text-only runs below do not execute. Measured on 8 x RTX 5060 Ti with the KDA capability guard lifted locally for the run; the lift is not part of the branch.

--- PASTE BEGIN ---

### Summary

Kimi K3 declared no `torch_remat` regions, so under [RegionAC](https://github.com/pytorch/torchtitan/blob/main/docs/remat.md) every op of a block was recomputed, including the KDA and MLA kernels. This PR declares regions at Kimi K3's call sites, the way `GQAttention` and `FeedForward` already do, so a save policy can keep the attention activations and recompute only the MoE or feed-forward:

```text
activation-checkpoint:region --activation-checkpoint.save-regions '*attention.*'
```

A recipe can list the patterns instead: `RegionAC.Config(save_regions=["attention.*", "delta_attention.*"])`.

No new config field and no change to `parallelize_kimi_k3`: RegionAC applies through the existing `ac_policy.apply(model)`. Outside a `torch_remat` checkpoint the regions do not change execution.

### Design

- MLA (`KimiMLAAttention`): `attention.q` (query down projection, norm, up projection), `attention.kv` (key/value latent, norm, up projection, and the shared rope key broadcast to the heads), `attention.inner_attention`, `attention.gate`, `attention.wo`.
- KDA (`KDA`), in main's op order: `delta_attention.forget`, `delta_attention.beta`, `delta_attention.qkv`, `delta_attention.inner_kda` (short convolution plus the Attention Gym kernel), `delta_attention.output_gate`, `delta_attention.output_norm`, `delta_attention.output_proj`.
- Block (`KimiK3TransformerBlock`): `attention_res` and `ffn_res`, the two attention-residual computations. Their math upcasts the whole block stack to fp32, so leaving them out of a save policy recomputes those intermediates instead of keeping them per layer.
- Behaviour change: the attention-residual math is now recomputed only under an enclosing RegionAC checkpoint; under selective AC or no AC it saves its fp32 intermediates as main does, since the earlier always-on checkpoint wrapper is gone.
- The MoE needs nothing beyond core: the router's routing decision is already a saved region, and the shared experts and dense feed-forward use `FeedForward`'s `w13` / `w2` regions.
- `remat.recompute_needs_tensor` sits before every bare consumer of a region output (the gated attention output, the reshaped beta, the norm inputs after the residual, the attention outputs read by the block's residual sum), per the consumer-side rule in the remat doc.

### Results

Kimi K3 debug model (24 layers plus an 8-block vision tower), dp1, bf16, `seed=42`, deterministic, 2048 tokens per step in 512-token micro-batches, 10 steps, one GPU, one inductor cache shared by every cell and warmed by a 1-step run of each. The source measured here is `c9d8ed39e` on `1c7ab8089`, the same diff as the branch's first commit (the second only adds the test).

```bash
PYTHONPATH=<attention-gym main>:. torchrun --nproc_per_node=1 -m torchtitan.train \
  --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
  --training.steps 10 --training.num-tokens-per-train-step 2048 \
  --training.num-tokens-per-microbatch-per-dp-rank 512 \
  activation-checkpoint:region --activation-checkpoint.save-regions '*attention.*'
```

| tree | activation checkpointing | step 1 loss / grad norm | step 10 loss / grad norm | loss and grad norm, steps 1 to 10 | peak memory | tps (mean of steps 6 to 10) |
| --- | --- | --- | --- | --- | ---: | ---: |
| main `1c7ab8089` | none | `12.63048` / `20.6250` | `3.64293` / `4.9062` | reference | 14.61 GiB | 858 |
| this PR | none | same | same | bitwise | 14.61 GiB | 849 |
| this PR | selective (the flavor default) | same | same | bitwise | 12.72 GiB | 505 |
| this PR | region, `save-regions '*attention.*'` | same | same | bitwise | 13.18 GiB | 648 |
| this PR | full | same | same | bitwise | 12.52 GiB | 637 |

With activation checkpointing off the PR is bitwise with main over the ten steps: the KDA projections keep main's op order (forget gate, beta, then query / key / value), so the regions add no numeric change. The log confirms RegionAC wrapped all 24 decoder blocks and the 8 vision blocks with the `*attention.*` pattern. Keeping the MLA and KDA activations and recomputing the MoE / feed-forward and the residual math costs 0.46 GiB over selective AC and runs 28% faster than it (648 against 505 tokens per second), because the attention kernels are no longer re-run in backward. `*attention.*` matches both `attention.*` (MLA) and `delta_attention.*` (KDA) and neither residual region; the CLI takes one pattern per option, so the single glob is the command-line spelling of that policy.

### Tests

`tests/unit_tests/cpu/test_kimi_k3_remat_regions.py` builds a KDA block and an MLA block from the debug model's layer configs, with CPU stand-ins for the CUDA-only kernels: every region is traced under its block-relative name; for each save policy (`[]`, the attention globs, the two residual regions, `["*"]`) the unsaved regions are recomputed exactly once and the saved ones are not; outputs and every gradient stay bitwise equal to the same blocks without activation checkpointing; applying RegionAC leaves the state dict unchanged.

### Changed files

```text
torchtitan/models/kimi_k3/model.py                    +59/-19  MLA and attention-residual regions
torchtitan/models/kimi_k3/kda.py                      +55/-12  KDA regions
tests/unit_tests/cpu/test_kimi_k3_remat_regions.py    +184/-0  region names, recompute counts, bitwise gradients
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
