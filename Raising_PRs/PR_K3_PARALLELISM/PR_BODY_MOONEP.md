# PR title: [Kimi K3] MoonEP as a MoE comm backend, on the standard dispatcher seam

Branch `k3_moonep_seam` = [`cc46bde23`](https://github.com/QIU023/torchtitan/commit/cc46bde23), four commits on main `b21f7d43e` (2026-09-14). Draft until the package is public on the CI boxes and the H100 SXM run is done; the CPU tests need neither the package nor a GPU. Body in the #4577 format (2026-09-14).

Notes for filing:
- Fixed in `cc46bde23` from the 2026-09-14 audit of `610f721bf`: the mesh check is now `dp_shard * cp * tp == ep`, one prefetch slot count (dispatcher config, read by the experts at attach), the expert GEMMs through `GroupedExperts._grouped_mm`, one combine call in the dispatch backward, the pass-through `wire_meshes` override gone, stale docstrings rewritten. Docstring size not re-measured.
- Commit `8fef1aa5f`'s message says two EP-group barriers per MoE layer per step; the code has three (one in prefetch, two in reduce). The body states three.
- The 2 x RTX 5060 Ti bitwise cells were measured on `610f721bf` (standard path and the EP=1 fallback), not rerun on `cc46bde23`. An earlier version of the table backend passed the four H100 checks on 2026-08-28; this one is that version ported onto this base.
- `PASS-COUNT`: fill from a run on `cc46bde23` before filing (12 test functions in the two new test files).

--- PASTE BEGIN ---

## Summary

Add MoonEP (MoonshotAI/MoonEP, the balanced EP transport of the Kimi K3 report) as a Kimi K3 MoE comm backend, selected with `model_registry(..., moe_comm_backend="moonep")`.

- `MoonEPTokenDispatcher` (`kimi_k3/moon_ep_dispatcher.py`): a `BaseEPTokenDispatcher` subclass whose dispatch and combine run MoonEP's kernels on a persistent buffer allocated from `wire_meshes` on the EP group.
- `MoonEPGroupedExperts` (`kimi_k3/moon_ep_experts.py`): a `GroupedExperts` subclass that computes over MoonEP's `[E + B]` tables (the `E` home experts plus `B` prefetch slots) through `GroupedExperts._grouped_mm`.
- `update_ep_token_dispatcher_config` (`models/common/token_dispatcher.py`): fills the static token capacity of every EP dispatcher config that declares `static_token_capacity`, instead of naming DeepEP and HybridEP; both declare it, so their behaviour is unchanged.
- Add CPU tests: an in-process double of the MoonEP buffer and tables (ranks as threads) checked against a dense reference with a duplicated expert, forward and gradients, and the core capacity fill.

## Design

The buffer is sized by the latent width, since the routed experts consume the stream after `routed_down`, and by the static per-rank token count that core now fills after CP and TP have sharded the token axis. Dispatch and combine are autograd Functions whose backward is the other kernel on the same plan; routing weights are applied on the torchtitan side, so the router trains through the same path as with the standard dispatcher. MoonEP's planner picks the experts to copy (`plan.experts_to_copy`, recomputed per dispatch); the experts fill the slots from the home ranks before the grouped GEMM and send the slot gradients home after the backward, over MoonEP's public `create_nvl_single_owner_tensor` (every rank maps every other rank's expert chunk and slot gradients), with a barrier on the EP group between the writes and the remote reads.

The transport stays in the model folder, like fla, and the package is imported only when an EP mesh exists; with no EP mesh both classes are their parents, so a flavor carrying the config still runs unsharded. The first version keeps expert parameters whole per EP rank and refuses `dp_replicate` and any mesh with `dp_shard * cp * tp != ep`.

Requirements and cost: Hopper or newer with NVSwitch. Every MoonEP buffer builds a multicast tensor ([`moonep/api.py#L362`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/api.py#L362)) and asserts multicast support ([`moonep/buffer.py#L299`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/buffer.py#L299)), the planner writes with `multimem.st` ([`moonep/planning.py#L149-L156`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/planning.py#L149-L156)), and prefetch and the dispatch epilogue use TMA ([`moonep/prefetch.py#L129-L208`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/prefetch.py#L129-L208), [`moonep/dispatch_epilogue.py#L197-L227`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/dispatch_epilogue.py#L197-L227)), so A100, H100 NVL pairs and PCIe cards cannot run it. Every rank holds `[E + B, in, out]` bf16 weight tables and fp32 gradient tables for the three projections, for all `E` experts; each MoE layer costs three EP-group barriers and two host syncs (`plan.experts_to_copy.tolist()`) per step. MoonEP's fused `prefetch_weight` / `reduce_grad` can replace the copies once the expert chunks are one contiguous VMM range. Dispatch and combine are not `torch.library` ops like core's DeepEP / HybridEP, so model compile breaks the graph at every dispatch.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py -q` (`PASS-COUNT`): spec selection and latent sizing, the EP=1 local fallback, the import guard, the mesh check, the two-rank fake world against a dense reference with a duplicated expert; the capacity fill after CP and TP, the EP=1 refusal, the divisibility check, the local fallback, a backend without a static capacity.
- `pytest tests/unit_tests/cpu/test_inference_moe.py tests/unit_tests/cpu/test_config_manager.py` and the Kimi K3 CPU tests (`PASS-COUNT`); ufmt and pyrefly on the changed files (`PASS-COUNT`).
- 2 x RTX 5060 Ti, Kimi K3 debug model, seed 42, deterministic, 3 steps, one warmed inductor cache per mesh (no NVSwitch, so the transport itself does not run): with the standard backend, dp1 and dp2 x ep2 are bitwise with main; moonep at EP=1 (the local fallback) is bitwise with the standard backend.
- To run on 2 x H100 SXM, moonep `2bd860b`: moonep's own two-rank tests, ep2 x fsdp2 with moonep for 3 steps, the per-parameter gradient comparison against the standard dispatcher on the same seed, and a forced-hot routing run that populates a slot.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
