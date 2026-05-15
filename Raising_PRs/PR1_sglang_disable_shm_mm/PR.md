# PR #1 — `SGLANG_DISABLE_SHM_MM` env to force CPU mm transport

**Target repo**: `sgl-project/sglang`
**Target file**: `python/sglang/srt/managers/tokenizer_manager.py` (function `_determine_tensor_transport_mode`)
**Fork reference**: commit `74083ffae` on `attention_residual_inference` branch.
**Effort**: ~30 min including PR description + a docstring sentence.
**Risk**: low (env-gated, default behavior unchanged).

---

## Suggested PR title

> Add `SGLANG_DISABLE_SHM_MM` env to force CPU multimodal transport in single-node setups

---

## Suggested PR body

### Summary

Adds an opt-in environment variable, `SGLANG_DISABLE_SHM_MM=1`, that
forces the multimodal tensor-transport path in `tokenizer_manager` to
fall back from `cuda_ipc` / POSIX-SHM to the default inline-pickle
transport. Default behavior is unchanged.

### Motivation

The default POSIX-SHM bridge for multimodal payloads
(`/psm_<random>`) races against container / actor lifecycles in
environments where the SGLang server is spawned inside a
parent-managed process tree:

- Monarch-style mesh actors: the spawning actor's `Provisioner`
  unlinks `/psm_*` before the scheduler subprocess opens it.
- Ray-style actor groups with SHM cleanup on `__del__` racing the
  ipc handle handoff.
- SLURM job arrays with shared `/dev/shm` quota where one job's
  cleanup nukes another job's psm objects.

Symptom is a hard `FileNotFoundError: '/psm_xxx'` deep inside the
multimodal dispatch path, with no recovery — the SGLang Engine
can't boot until the operator manually scrubs `/dev/shm` and rebuilds
the process tree.

The cross-node code path already takes a different transport branch
(`server_args.dist_init_addr` set) since SHM is meaningless across
nodes. This PR exposes the same opt-out for single-node setups that
hit the SHM lifecycle race.

### Patch

```python
# python/sglang/srt/managers/tokenizer_manager.py

def _determine_tensor_transport_mode(server_args) -> TensorTransportMode:
    if os.environ.get("SGLANG_DISABLE_SHM_MM", "0") == "1":
        return "default"  # inline pickle, lifecycle-safe
    if server_args.dist_init_addr:
        return "default"
    return "cuda_ipc"
```

Plus a one-paragraph note in the multimodal serving docs:

> Set `SGLANG_DISABLE_SHM_MM=1` if the engine boots inside an actor /
> RPC framework that manages its own process tree (Monarch, Ray
> placement groups, SLURM job arrays). The default `cuda_ipc` /
> POSIX-SHM transport is lifecycle-unsafe in those environments.

### Why low risk

- Default behavior unchanged.
- Already a precedent: the cross-node `dist_init_addr` branch returns
  `default`. This PR just extends that opt-out to single-node use.
- No code path other than the transport selection touches `os.environ`
  for this var.

### Test plan

1. Default smoke (`SGLANG_DISABLE_SHM_MM` unset) — existing multimodal
   tests stay green; transport mode is `cuda_ipc` as before.
2. With env set — repro a single-node Monarch actor mesh that wraps
   the SGLang Engine; confirm boot succeeds and inline pickle
   transport is selected.

### Reference downstream usage

Used in our research fork (`QIU023/torchtitan_attention_residual`)
to unblock GRPO RL rollouts driven by torchtitan
`experiments/rl/SGLangGenerator` under a Monarch actor mesh. Without
the env, the GRPO smoke launcher hits the SHM lifecycle race on the
first rollout call.

---

## Filing checklist

- [ ] Fork branch up to date with sglang `main`.
- [ ] Single-commit PR titled per above.
- [ ] Description includes the symptom + repro context (Monarch
      / Ray / SLURM).
- [ ] Test plan checked locally on both default and opt-in paths.
- [ ] Link to the discussion in our fork's
      `phase11_rlhf_grpo_infra/UPSTREAM_PR_LIST.md` so reviewers see the broader
      context if curious.
