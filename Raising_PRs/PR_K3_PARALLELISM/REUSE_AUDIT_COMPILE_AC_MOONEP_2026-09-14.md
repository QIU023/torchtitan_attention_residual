# Reuse audit: compile, AC reuse, MoonEP (2026-09-14)

Against upstream main `1c7ab8089`, per the CLAUDE.md abstraction and reuse-check rules. Branches: `k3_compile_blocks` = `5ee84f0c4`, `k3_ac_reuse_attention` = `6ef880995`, `k3_moonep_seam` = `3756a97d5` (all on `ac10ca48f`, 45 behind main).

## compile (`k3_compile_blocks`)

New defs: `_kernels_carved_out` (module global), `_carve_kernels_out_of_dynamo`, `_apply_compile_kimi_k3`; core `apply_compile(fullgraph=)`.

- `fullgraph` parameter on core `apply_compile`: no existing seam on main (every model passes through the hard-coded `fullgraph=True`; gpt_oss raises the recompile limit instead). No house precedent contradicts it, but the body must name the concrete op that breaks the graph. Main has no compiled Qwen3.5 cell, so "GDN compiles under fullgraph" is not established either; attn_gym main itself marks its cute delta-rule kernels `@torch.compiler.disable` (`attn_gym/linear/_delta_rule/cute/affine_summary_{fwd,rev}.py`). Verify which call in `KDAKernel.forward` breaks the graph on the box before filing.
- Runtime monkeypatch guarded by a module global: replace with a static `@torch.compiler.disable(recursive=True)` on `KDAKernel.forward` in `kda.py` (the idiom `distributed/compile.py:24-29` already uses). Removes the global and the helper.
- House-aligned alternative if the maintainers refuse the core parameter: wrap the kernel as `torch.library.custom_op` with `register_fake` / `register_autograd` (pattern: `qwen3_5/gdn.py:83-202`, `distributed/deepep/hybridep.py`, `overrides/fused_mla.py`), which keeps `fullgraph=True` and needs no core change. More work; hold in reserve.

## AC reuse (`k3_ac_reuse_attention`)

New defs: `_attention_residual_math`, `_apply_attention_residual` (checkpoint wrapper), `_apply_ac_outside_attention`, config flag `ac_reuse_attention`.

- Finding (major): main now has `RegionAC` (`distributed/activation_checkpoint.py:302-373`, #4526 stack): models declare `remat.region(fn, self.remat_region_name(...), recompute=self.remat_should_recompute(...))` at their call sites and the user picks `save_regions` globs; everything else in the block is recomputed. "Keep attention, recompute the MoE/FFN" is exactly a save policy over declared regions. The branch re-implements it as a model flag plus a function that calls the private `ActivationCheckpointing._wrap_block` on `layers.<i>.moe|feed_forward`. Upstream shape: declare K3 regions (MLA qkv / inner attention / wo like `models/common/attention.py:911-948`, the KDA kernel, the latent MoE, the attention-residual math) and drop the flag and `_apply_ac_outside_attention`.
- Finding: the attention-residual wrapper calls `torch.utils.checkpoint` directly; no model on main does (grep empty). As a `remat.region` with `recompute=...` it composes with RegionAC instead of nesting a second recompute inside the block's.
- Rebase fix: `from torchtitan.tools.logging import logger` no longer exists (#4628).

## MoonEP (`k3_moonep_seam`)

New defs: `_import_moonep`, `_MoonEPDispatch`, `_MoonEPCombine`, `MoonEPDispatchMetadata`, `MoonEPTokenDispatcher(BaseEPTokenDispatcher)`, `MoonEPTableBackend`, `_grouped_mm`, `_ep_coords`, `_MoonEPExpertFunction`, `MoonEPGroupedExperts`, `check_moonep_mesh`, `MoonEPTableBackendNVLink`, the model-config loop filling `num_max_tokens_per_rank`.

- Blocking (not reuse): `MoonEPTableBackendNVLink.alloc_weight_table` / `alloc_grad_table` raise `NotImplementedError` on the branch and on the integration tree. The seam commit (and the integration's `5e092d63d`) came from `599077a00`, the pre-box version; the allocator proven on 2 x H100 on 08-28 is `f09448ce8` + `f7c434b92` on the fork's `origin/main` line and was never ported. Port those two before any hardware run; without them ep > 1 cannot run.
- Finding (reuse): the K3 model-config loop recomputes `num_tokens_per_microbatch_per_dp_rank // (cp * tp)` for MoonEP, which core `update_ep_token_dispatcher_config` (`models/common/token_dispatcher.py:988`) already does for DeepEP / HybridEP, including the divisibility check the loop lacks. Generalize core's function to any EP dispatcher config that declares a static token capacity (MoonEP keeps its EP=1 local fallback, so the EP=1 refusal must be per backend), and delete the loop.
- Finding (minor): module-level `_grouped_mm(A, B_t, offs)` with a CPU loop. The table orientation is MoonEP's `[row, in, out]`, so `GroupedExperts._grouped_mm(weight_EOI=...)` does not fit directly; the CPU loop exists only for the fake-world test and belongs with the test double.
- Finding (minor): core's DeepEP / HybridEP register their dispatch / combine as `torch.library` ops (compile-traceable); MoonEP's are `autograd.Function`s. Fine for eager; note it in Limitations, since compile x moonep would graph-break.
- OK: dispatcher on the `BaseEPTokenDispatcher` seam with `wire_meshes` / `init_buffer`; experts on `GroupedExperts` (the integration port; the branch still names the removed `KimiGroupedExperts` and must take that port); `MoonEPDispatchMetadata` could be core's `EPDispatchMetadata(state=...)` but a typed record is defensible.

## Hardware note for MoonEP

Hopper or newer with NVSwitch only: `moonep/api.py:362` builds a multicast tensor for every Buffer, `moonep/buffer.py:299` asserts multicast support, the planner writes with `multimem.st` (`moonep/planning.py:149-156`), and prefetch / dispatch use TMA (`moonep/prefetch.py`, `moonep/dispatch_epilogue.py`). A100, H100 NVL pairs and PCIe cards cannot run it.
