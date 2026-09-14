# PR title: [Kimi K3] MoonEP as a MoE comm backend, on the standard dispatcher seam

Branch `k3_moonep_seam` = `610f721bf` (three commits on main `1c7ab8089`, 2026-09-14). Draft until the package is public on the CI boxes and the hardware run below is done; the CPU tests need neither the package nor a GPU.

--- PASTE BEGIN ---

## Summary

Adds `"moonep"` to the comm backends Kimi K3 accepts through `model_registry(..., moe_comm_backend=...)`: MoonEP (MoonshotAI/MoonEP, the report's balanced EP transport) as a `BaseEPTokenDispatcher` subclass plus the expert module that computes over its tables. The transport stays in the model folder, like fla, and the package is imported only when an EP mesh exists. One small core change lets core fill the static token capacity of any persistent EP backend instead of naming DeepEP and HybridEP.

- Dispatcher ([10defb02e](https://github.com/QIU023/torchtitan/commit/10defb02e0dd680af908cb1d4396e4d745ae9952)): the persistent buffer is allocated once from `wire_meshes` on the EP group, sized by the latent width (the routed experts consume the stream after `routed_down`) and by the static per-rank token count. Dispatch and combine are autograd Functions whose backward is the other kernel on the same plan; routing weights are applied on the torchtitan side, so the router trains through the same path as with the standard dispatcher.
- Experts ([10defb02e](https://github.com/QIU023/torchtitan/commit/10defb02e0dd680af908cb1d4396e4d745ae9952), [3fc1386f1](https://github.com/QIU023/torchtitan/commit/3fc1386f1a46f8cb8d766f5e523ba0599bab5ce4)): `MoonEPGroupedExperts` subclasses core `GroupedExperts` (same parameters, same `activation_fn`) and computes over MoonEP's `[E + B]` tables: the `E` home experts plus `B` prefetch slots. The table backend fills the slots with the experts `plan.experts_to_copy` names and returns their gradients to the home ranks, over MoonEP's public `create_nvl_single_owner_tensor` (every rank owns an NVLink-mapped copy of its expert chunk and of its slot grads and maps every other rank's), with a barrier on the EP group between the writes and the remote reads.
- Core ([610f721bf](https://github.com/QIU023/torchtitan/commit/610f721bf6ba02fad7ef2ffd3c23ec25fb6c0d6d)): EP dispatcher configs declare `static_token_capacity` (and `ep1_local_fallback` when they also run without EP); `update_ep_token_dispatcher_config` fills `num_max_tokens_per_rank` for every config that declares it, with its existing divisibility and capacity checks. DeepEP and HybridEP declare the first flag, so their behaviour is unchanged; MoonEP declares both.
- With no EP mesh both MoonEP classes are their parents, so a flavor carrying the config still runs unsharded, and the standard backend's path is untouched.

## Changed files

```text
torchtitan/models/common/token_dispatcher.py            +18/-9   static_token_capacity / ep1_local_fallback, filled by update_ep_token_dispatcher_config
torchtitan/models/kimi_k3/__init__.py                   +26/-10  "moonep" selects the MoonEP dispatcher and experts in _latent_moe_config
torchtitan/models/kimi_k3/moe.py                        +23/-0   KimiLatentMoE.parallelize attaches the expert side and checks the mesh
torchtitan/models/kimi_k3/moon_ep_dispatcher.py         +331/-0  the dispatcher
torchtitan/models/kimi_k3/moon_ep_experts.py            +384/-0  the experts and the NVLink table backend
tests/unit_tests/cpu/kimi_k3_moonep_fake.py             +305/-0  in-process Buffer / table double, ranks as threads
tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py +203/-0  spec, EP=1 fallback, import guard, two-rank dense reference
tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py +87/-0 core capacity fill
```

## Limitations

- Hardware: Hopper or newer with NVSwitch. Every MoonEP `Buffer` builds a multicast tensor ([`moonep/api.py#L362`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/api.py#L362)) and asserts multicast support ([`moonep/buffer.py#L299`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/buffer.py#L299)); the planner writes with `multimem.st` ([`moonep/planning.py#L149-L156`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/planning.py#L149-L156)); prefetch and the dispatch epilogue use TMA ([`moonep/prefetch.py#L129-L208`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/prefetch.py#L129-L208), [`moonep/dispatch_epilogue.py#L197-L227`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/dispatch_epilogue.py#L197-L227)). A100, H100 NVL pairs and PCIe cards cannot run it; without the package the import guard raises at `parallelize` with the package name.
- The first version keeps expert parameters whole per EP rank (`dp_shard == ep`, no `dp_replicate`) and refuses other meshes at `parallelize`. TP x moonep is not exercised.
- The table backend does prefetch and slot-grad reduce with copies and two EP-group barriers per MoE layer per step, and `plan.experts_to_copy.tolist()` is a device sync per layer; MoonEP's fused `prefetch_weight` / `reduce_grad` can replace both once the expert chunks are exposed as one contiguous VMM range.
- Dispatch and combine are `autograd.Function`s, not `torch.library` ops like core's DeepEP / HybridEP, so model compile with moonep breaks the graph at every dispatch.
- The planner's copy decision is moonep's (stateless, recomputed per dispatch from the all-gathered `tokens_per_expert`); the integration passes `plan=None` and adds no policy knobs.

## Tests

```text
pytest tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py
```

Result: 11 passed. The MoonEP tests cover spec selection and latent sizing, the EP=1 local round trip, the import guard, and the two-rank fake world (`kimi_k3_moonep_fake.py` implements the Buffer and table backend surface in-process, ranks as threads, a test-chosen duplication map) against a dense reference with a duplicated expert, forward and gradients. The capacity tests cover the fill after CP and TP, the EP=1 refusal, the divisibility check, the local fallback and a backend without a static capacity. `test_inference_moe.py`, `test_config_manager.py` and the Kimi K3 CPU tests pass (50). ufmt is clean on every changed file; pyrefly reports no error in a changed file.

## Results

Kimi K3 debug model, `seed=42`, deterministic init, 8192 tokens per step (256 per micro-batch per dp rank), 3 steps, 2 x RTX 5060 Ti (SM120; the KDA kernels run on Attention Gym's portable Triton path with the capability check relaxed locally for the run), one warmed inductor cache per mesh shared by every cell of that mesh. main `1c7ab8089` is the reference.

| cell | backend | step 1 loss / grad norm | step 2 | step 3 |
| --- | --- | ---: | ---: | ---: |
| dp1, main | standard | `12.54770` / `15.1250` | `9.89441` / `14.4375` | `7.69925` / `8.0000` |
| dp1, this branch | standard | bitwise | bitwise | bitwise |
| dp1, this branch | moonep (EP=1 local fallback) | bitwise | bitwise | bitwise |
| dp2 x ep2, main | standard | `12.40228` / `14.8125` | `9.58659` / `13.2500` | `7.52933` / `10.3750` |
| dp2 x ep2, this branch | standard | bitwise | bitwise | bitwise |

No moonep package and no NVSwitch on this box, so the transport itself is not exercised here; the two-rank fake world in the tests is the functional check of the dispatch, slot, combine and reduce path.

Still to run on 2 x H100 SXM (NVSwitch), moonep master `2bd860b`: moonep's own two-rank tests, ep2 x fsdp2 with moonep for 3 steps, the per-parameter gradient comparison against the standard dispatcher on the same seed, and a forced-hot routing run that populates a slot. An earlier version of this code passed all four on 2 x H100 on 2026-08-28; the table backend above is that version's, ported onto this base.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
