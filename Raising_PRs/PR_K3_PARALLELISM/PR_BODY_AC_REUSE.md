# PR title: [Kimi K3] Declare torch_remat regions so RegionAC can keep attention and recompute the MoE

Fork branch `k3_ac_reuse_attention` = [`bc696be15`](https://github.com/QIU023/torchtitan/commit/bc696be15eca3fc26559c9aab76f7adde8c993a1) (two commits on main `b21f7d43e`, independent of the parallelism PRs). Replaces the earlier draft of this branch (a model flag, a private wrap of the MoE / feed-forward, a direct `torch.utils.checkpoint` around the attention residual) with region declarations on main's RegionAC. Body in the #4577 format (2026-09-14).

Notes for filing:
- Measured on `c9d8ed39e` on `1c7ab8089`, the same diff as the branch's first commit; that object is not on the local fork, so the equivalence is the box's statement. The newer main commit `b21f7d43e` (#4611) only adds vision-encoder regions, which the text-only runs do not execute.
- One RTX 5060 Ti of the 8-GPU box, KDA capability guard lifted locally for the run (not part of the branch). Raw logs are not in the logbook.
- `PASS-COUNT`: fill from a run on `bc696be15` before filing (the same test read 3 / 3 on the integration tree).

--- PASTE BEGIN ---

## Summary

Declare `torch_remat` regions in Kimi K3 so [RegionAC](https://github.com/pytorch/torchtitan/blob/main/docs/remat.md) can keep the attention activations and recompute only the MoE or feed-forward; without them RegionAC recomputes every op of a block, the KDA and MLA kernels included.

- `KimiMLAAttention` (`model.py`): `attention.q`, `attention.kv`, `attention.inner_attention`, `attention.gate`, `attention.wo`.
- `KDA` (`kda.py`), in main's op order: `delta_attention.forget`, `.beta`, `.qkv`, `.inner_kda`, `.output_gate`, `.output_norm`, `.output_proj`.
- `KimiK3TransformerBlock` (`model.py`): `attention_res` and `ffn_res`, the two attention-residual computations.
- Add a CPU test of the region names, the recompute counts per save policy, and bitwise outputs and gradients.

## Design

The regions are declared at the call sites with `remat.region(fn, self.remat_region_name(...), recompute=self.remat_should_recompute(...))`, the way `GQAttention` and `FeedForward` do, and `remat.recompute_needs_tensor` precedes every bare consumer of a region output, per the remat doc. The MoE needs nothing new: the router's decision is already a saved region, and the shared experts and dense feed-forward use `FeedForward`'s `w13` / `w2` regions. RegionAC applies through the existing `ac_policy.apply(model)`, so there is no new config field and no change to `parallelize_kimi_k3`; outside a `torch_remat` checkpoint the regions do not change execution.

The attention-residual math upcasts the whole block stack to fp32, so leaving its two regions out of a save policy recomputes those intermediates instead of keeping them per layer. On the command line the policy is `activation-checkpoint:region --activation-checkpoint.save-regions '*attention.*'` (one pattern per option; `*attention.*` matches `attention.*` and `delta_attention.*` and neither residual region); a recipe can write `RegionAC.Config(save_regions=["attention.*", "delta_attention.*"])`.

## Results

Kimi K3 debug model (24 layers plus an 8-block vision tower, text-only data), dp1, bf16, seed 42, deterministic, 2048 tokens per step in 512-token micro-batches, 10 steps, one GPU, one inductor cache shared by every cell and warmed by a one-step run of each.

```bash
PYTHONPATH=<attention-gym main>:. torchrun --nproc_per_node=1 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.steps 10 --training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 512 activation-checkpoint:region --activation-checkpoint.save-regions '*attention.*'
```

| tree | activation checkpointing | step 1 loss / grad norm | step 10 loss / grad norm | loss and grad norm, steps 1 to 10 | peak memory | tps (mean of steps 6 to 10) |
| --- | --- | --- | --- | --- | ---: | ---: |
| main `1c7ab8089` | none | `12.63048` / `20.6250` | `3.64293` / `4.9062` | reference | 14.61 GiB | 858 |
| this PR | none | same | same | bitwise | 14.61 GiB | 849 |
| this PR | selective (the flavor default) | same | same | bitwise | 12.72 GiB | 505 |
| this PR | region, `save-regions '*attention.*'` | same | same | bitwise | 13.18 GiB | 648 |
| this PR | full | same | same | bitwise | 12.52 GiB | 637 |

Keeping the MLA and KDA activations costs 0.46 GiB over selective AC and runs 28% faster (648 against 505 tokens per second), because the attention kernels are no longer re-run in backward.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_remat_regions.py -q` (`PASS-COUNT`): a KDA block and an MLA block from the debug model's layer configs with CPU stand-ins for the CUDA-only kernels; every region traced under its block-relative name, each save policy recomputes exactly the regions it does not keep, outputs and every gradient bitwise equal to the blocks without activation checkpointing, the state dict unchanged by RegionAC.
- The GPU command above, once per row.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
