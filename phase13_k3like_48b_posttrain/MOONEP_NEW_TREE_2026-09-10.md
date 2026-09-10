# MoonEP on the new tree (2026-09-10)

Port of the MoonEP dispatcher and expert side (`k3_on_4025` commits d51e27b35, b4e104a6b, 7b2157c0d; draft body `Raising_PRs/PR_K3_PARALLELISM/PR_BODY_MOONEP.md`) onto `k3_int_20260910`.

## Where it lands

- main's seam is the one the draft already used: `model_registry(..., moe_comm_backend=...)` reaches `_latent_moe_config`, which hands every core backend to `make_token_dispatcher_config` and builds `MoonEPTokenDispatcher.Config` (sized by the latent width) plus `MoonEPGroupedExperts.Config` for `"moonep"`. Core's factory is untouched, so the optional dependency stays in the model folder like fla.
- Core's `update_ep_token_dispatcher_config` fills the static per-rank token count only for the backends it names and refuses EP=1 for them; MoonEP keeps the EP=1 local fallback, so the K3 model config fills the count itself (tokens per micro-batch per DP rank over CP x TP), as the historical commit did.
- `KimiLatentMoE.parallelize` attaches the expert side after the children are parallelized (the new tree's forward takes `**router_kwargs`; only the anchor moved).
- The tests moved to `tests/unit_tests/cpu/` (`kimi_k3_moonep_fake.py`, `test_kimi_k3_moon_ep_dispatcher.py`), which is what `pyproject.toml`'s `testpaths` collects; the model-folder `tests/` on main is not on that path.
- Pinned pyrefly (0.45.1) on the new tree: the autograd Functions carry override ignores, `combine` narrows its metadata argument by type instead of annotating the MoonEP class (the EP=1 fallback passes core's `LocalDispatchMetadata`), and the expert forward's parameter names now match the parent's.

## Evidence on this box

No `moonep` package and no NVLink here, so the CPU fake is the whole verification on this tree: 4/4 (`test_kimi_k3_moon_ep_dispatcher.py`: spec selection and latent sizing, EP=1 local round trip, import guard, the two-rank fake world against a dense reference with a duplicated expert, forward and gradients). The K3 CPU set stays green (83 passed). The H100 NVLink evidence in the draft body was taken on the old tree and is not re-run.
