# PR title: [Kimi K3] MoonEP as a MoE comm backend, on the standard dispatcher seam

Branch `k3_moonep_seam` = `3756a97d5` (one commit on main `ac10ca48f`). No stack: the seam it plugs into (`model_registry(..., moe_comm_backend=...)` reaching `_latent_moe_config`) is on main. Draft until the package is public on the CI boxes; the CPU tests need neither the package nor a GPU.

--- PASTE BEGIN ---

## Summary

Adds `"moonep"` to the comm backends Kimi K3 accepts through `model_registry(..., moe_comm_backend=...)`: MoonEP (MoonshotAI/MoonEP, the report's balanced EP transport) as a `BaseEPTokenDispatcher` subclass plus the expert module that computes over its tables. Everything stays in the model folder, like fla; core's dispatcher factory keeps no non-PyTorch dependency, and the package is imported only when an EP mesh exists.

- The dispatcher is written to the released moonep API: the persistent buffer is allocated once from `wire_meshes` on the EP group, sized by the latent width (the routed experts consume the stream after `routed_down`) and by the static per-rank token count, which the model config fills from the training shape after CP and TP/SP have sharded the token axis (core fills the same figure for its own persistent backends and does not know this one). Dispatch and combine are autograd Functions whose backward is the other kernel on the same plan; routing weights are applied on the torchtitan side, so the router trains through the same path as with the standard dispatcher.
- The expert side computes over moonep's `[E + B]` weight tables: the `E` home experts plus `B` prefetch slots that `prefetch_weight` fills before the grouped GEMM. The backward recomputes the expert forward with the tables as leaves, routes the local rows to the local parameters' gradients and the slot rows to the reduce buffer, and `reduce_grad` returns the duplicated experts' gradients to their home ranks.
- With no EP mesh both classes are their parents, so a flavor carrying the config still runs unsharded, and the standard backend's path is untouched.

## Implementation

`__init__.py`: `"moonep"` selects `MoonEPTokenDispatcher.Config` and `MoonEPGroupedExperts.Config` in `_latent_moe_config`, sized by `latent_dim` like the core dispatchers. `moe.py`: `KimiLatentMoE.parallelize` attaches the expert side once the dispatcher's plan and the EP mesh exist (the NVLink table backend on the EP mesh) and checks the mesh. `model.py`: the model config fills the dispatcher's `num_max_tokens_per_rank` from `num_tokens_per_microbatch_per_dp_rank` divided by the cp x tp shards. `moon_ep_dispatcher.py` and `moon_ep_experts.py` are the two new modules.

## Limitations

- The first version keeps expert parameters whole per EP rank (`dp_shard == ep`, no `dp_replicate`) and refuses other meshes at `parallelize`.
- NVLink-mapped GPUs and the moonep package are required for EP > 1; without the package the import guard raises at `parallelize` with the package name. TP x moonep is not exercised.
- The planner's copy decision is moonep's (stateless, recomputed per dispatch from the all-gathered `tokens_per_expert`); the integration passes `plan=None` and the per-step histogram, and does not add policy knobs.

## Tests

```text
pytest tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py tests/unit_tests/cpu/test_integration_test_definitions.py
```

Result: 17 passed (the four MoonEP tests: spec selection and latent sizing, the EP=1 local round trip, the import guard, and the two-rank fake world -- `kimi_k3_moonep_fake.py` implements the Buffer API surface in-process, ranks as threads, a test-chosen duplication map -- against a dense reference with a duplicated expert, forward and gradients). pre-commit passes on the touched files; `pyrefly check` reports the same error set as main `ac10ca48f` (the environment's pre-existing missing imports, no error in a touched file). The autograd Functions carry `bad-override` ignores that the pinned pyrefly 0.45.1 needs.

## Results

Kimi K3 debug model, `seed=42`, deterministic, one seed checkpoint for every cell, 4096 tokens per step (256 per micro-batch per dp rank), 3 steps, spmd_types, 8 x RTX 5060 Ti (SM120; the KDA kernels run on Attention Gym's portable Triton path). main `ac10ca48f` is the reference; the standard backend is untouched by this PR and reads bitwise; the `moonep` flavor at EP=1 is the local fallback and reads bitwise; at EP=2 without the package the run stops at `parallelize` with the import guard.

| cell | backend | step 1 loss / grad norm | step 2 | step 3 |
| --- | --- | ---: | ---: | ---: |
| dp1, main | standard | `12.53584` / `14.6250` | `9.49238` / `16.2500` | `6.60885` / `9.7500` |
| dp1, this branch | standard | bitwise | bitwise | bitwise |
| dp1, this branch | moonep (EP=1 fallback) | bitwise | bitwise | bitwise |
| dp2 x ep2, main | standard | `12.52733` / `15.0625` | `9.62503` / `14.2500` | `7.15118` / `12.1875` |
| dp2 x ep2, this branch | standard | bitwise | bitwise | bitwise |
| dp2 x ep2, this branch | moonep | refused at `parallelize` by the guard: `ImportError: MoonEP is not installed. It is an optional dependency, like DeepEP: install from https://github.com/MoonshotAI/MoonEP, and note that it requires NVLink-connected GPUs. Use another comm_backend on hardware without that topology.` | | |

No moonep package and no NVLink on this box, so the transport itself is not exercised here; the CPU fake world in the tests is the functional check of the dispatch / slot / combine / reduce path.

Earlier evidence with the package, on the tree this port comes from (2 x H100 80GB, NVLink NV18, moonep master `2bd860b`): moonep's own two-rank tests (planning, dispatch, combine, e2e, grad_reduce, prefetch) pass; ep2 x fsdp2 trains (12.47486 -> 9.19460 over three steps, deterministic seed); the per-parameter gradient probe against the standard dispatcher on the same seed shows the largest relative differences at 2.1-2.6e-1 on 1e-4-magnitude parameters with no parameter family deviating; with routing forced hot a slot populates and prefetch, the slot GEMM and reduce-grad all execute, at the balanced run's throughput (345 vs 347 tps). The dispatcher and expert modules are unchanged by the port; the seam code is the three small hunks above.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
