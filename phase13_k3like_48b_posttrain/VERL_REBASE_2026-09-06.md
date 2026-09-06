# verl: upstream state, Kimi K3 support, and the rebase of `kimi_k3_integration` (2026-09-06)

## Upstream (`verl-project/verl`, remote `upstream` added to the fork checkout)

`upstream/main` = `23af6a7a` (2026-09-04). Our fork branch `kimi_k3_integration` (`dbc08fd7`) sits on
`6a6242f3` (2026-07-20): 29 commits of ours over 190 of theirs, 575 upstream files changed, 8 of our
13 files among them.

**Kimi K3 support upstream: none.** No `kimi_k3` / KDA / AttnRes mention anywhere on main (verl,
docs, examples, tests); the only search hits are dependabot vLLM bumps whose changelogs name Kimi K3.
The torchtitan engine upstream (`verl/workers/engine/torchtitan/`) wires tp / pp / cp / ep degrees
into torchtitan's `ParallelDims` but has no pipeline schedule or CP input path of its own beyond
`prepare_context_parallel_input`; ours adds the pipeline bridge, the token budget and the CP packing.

Upstream changes since our base that touch the engine or its contract:

| PR | what | bearing on us |
| --- | --- | --- |
| #7324 (08-21) | sharded delta weight sync for the TorchTitan engine | new sync path in `transformer_impl.py`; our LoRA export / merged sync sits next to it, untested together |
| #7327 (+ #7376, #7413) | `.base_layer` resolved on the vLLM receiver for non-merged LoRA sync; trainer-side renaming and `megatron_peft_utils.add_base_layer_suffix` / `STACKED_PARAMS` removed | our three "name base_layer from what the rollout wraps" commits are superseded (skipped in the rebase); the engine still carries the older `_insert_base_layer_suffix(params, hf_names)`, which now double-renames against the receiver and must go |
| #7362 / #7374 | fsdpturbo engine added, mindspeedllm removed | none |
| #7212 (open) | SFT on TPU through the TorchTitan engine | a second consumer of the engine; watch for interface drift |
| #7127 (BREAKING) | uv integration | env / launch scripts |
| #7544 (BREAKING) | `grad_offload` removed from the config | our configs that set it must drop it |
| #7458 | deferred gradient sync configurable (fsdp) | none |
| #7408 | profiler post hook | none |
| #6555 | dynamic context parallel scheduling (megatron only) | not for the titan engine |
| flash-attn-less fallback, gloo composite backend | upstream now does both | two of our commits became redundant (taken from upstream) |

## The rebase (`kimi_k3_integration_rebased`, fork, `92dd4d21`; the original branch untouched, tag `pre_rebase_20260906`)

26 of 29 commits survive. Skipped: `ccb5b10a`, `16719071`, `bac16f4d` (the trainer-side base_layer
naming series, superseded by #7327). Conflicts resolved: `distributed.py` (gloo composite backend:
upstream's), `transformer_impl.py` `_get_data_parallel_mesh` (ours: the "batch" mesh excludes cp;
upstream's new "loss"-mesh lookup includes cp and would repeat the sampler crash we fixed),
`attention_utils.py` (upstream's torch fallback), `vllm_async_server.py` (ours: `VERL_VLLM_VERSION`
override and the reset-prefix-cache kwargs). Compiles.

Not verified by running: the engine's unit tests cannot run in either environment today.
- `venv_verl` carries spmd_types 0.2.1 while its torchtitan checkout (`/workspace/tt_4025/torchtitan`,
  `k3_on_4025` at `9f87e0891`, 2026-09-03 "run under the spmd_types backend") needs 0.2.5:
  `no_typecheck() got an unexpected keyword argument 'out_types'` at import. The last verl runs
  (09-02) predate that commit; the pair has been broken since.
- Under `/venv/main` with the new tree (`mm18_int`), the engine imports old-tree names:
  `torchtitan.models.kimi_k3.lora` and the `kimi_k3_debugmodel_gated_lora` flavor do not exist there.

## What the next verl step is

1. Point the engine at the new tree: `torchtitan.models.kimi_k3` on `mm18_int` / the review heads
   (LoRA through core's `LoRAConverter` and the `kimi_k3_debugmodel_lora` flavor, not
   `kimi_k3.lora`), re-derive the rl flavors (`rl_vit1` had DEP; the new tree has none) and rerun the
   09-02 cells (dp1, cp2, pp2) for the gap numbers.
2. Drop the trainer-side base_layer renaming and let #7327's receiver resolve names; check K3's
   fused projections (`in_proj_qkvgfab`, the AttnRes projections, the router) against the receiver's
   resolver, which is where our `_K3_STACKED_PARAMS` knowledge now belongs if anything is missing.
3. Reconcile our merged LoRA sync with #7324's delta-sharded sync (same file).
4. Env: either pin `venv_verl`'s torchtitan to a pre-`9f87e0891` state to re-verify the old cells, or
   rebuild it on the new tree with spmd_types 0.2.5 (the main venv's stack).
