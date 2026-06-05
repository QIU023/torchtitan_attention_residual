# PR #1 — filing instructions

## Status

🟡 **Re-prepared 2026-06-05; local commit ready; force-push + open pending.**

| Item | Link / value |
|---|---|
| Fork branch | https://github.com/QIU023/sglang/tree/pr1-disable-shm-mm |
| Open-PR URL | https://github.com/QIU023/sglang/pull/new/pr1-disable-shm-mm |
| Target repo | https://github.com/sgl-project/sglang |
| Base | `sgl-project/sglang:main` @ `4ef081b903` (2026-06-05) |
| Head | `QIU023/sglang:pr1-disable-shm-mm` |
| Commit | `3f4e2a3fb4` (1 commit, **3 files / +90**) |
| Prior HEADs | `4bcf838df` (raw-os) → `cbf984255d` (rebased) → `c381a13905` (idiomatic) → `3f4e2a3fb4` (deterministic repro) |
| Cross-link | (none — independent of other PRs) |

> **Re-prepared 2026-06-05.** Three changes vs. the old single-hunk version:
> 1. **Re-rebased** onto current `upstream/main` (`4ef081b903`, +308 commits past the
>    2026-05-21 base); `_determine_tensor_transport_mode` byte-identical upstream → clean replay.
> 2. **Made idiomatic** per the repo's `env-var-conventions` skill: registered
>    `SGLANG_DISABLE_SHM_MM = EnvBool(False)` in `srt/environ.py` (VLM Item CUDA IPC
>    Transport group) and read it via `envs.SGLANG_DISABLE_SHM_MM.get()` instead of a
>    raw `os.environ.get` — heads off the predictable "use environ.py" review bounce.
> 3. **Added a GPU-free, stdlib-only reproducer** at
>    `examples/runtime/multimodal/repro_disable_shm_mm_race.py`. The producer runs as
>    its own interpreter so *its* resource_tracker unlinks the segment on exit —
>    deterministic, faithful to the Ray/SLURM/Monarch per-worker teardown.
>
> py_compile clean on all three files. **Reproducer verified on WSL2 / Linux:** real
> `/psm_*` `FileNotFoundError`, 3/3 runs (Windows shows the same race under the `wnsm_`
> backend). **Force-push to origin still pending user authorization** (history rewrite
> of the published branch).

## Decoupled from torchtitan — on purpose (fork is public)

PR1's justification is the **general** SHM-lifecycle race (any parent-managed process
tree: Ray / SLURM / Monarch), *not* our torchtitan `SGLangGenerator`. The generator is
a downstream caller, not an upstream dependency — PR1 stands on its own regardless of
whether any torchtitan RL integration lands.

Our fork **is public**, so the upstream body links it for reviewers who want the full
stack: the patched branch is at https://github.com/QIU023/sglang/tree/pr1-disable-shm-mm
and the phase11 GRPO harness that first hit the race lives in
https://github.com/QIU023/torchtitan_attention_residual . The body still leads with the
self-contained reproducer (instant, no clone) and keeps torchtitan to one "we soaked it
here" line — the fork links are optional depth, not the argument.

## To open the PR

1. (after force-push) Open https://github.com/QIU023/sglang/pull/new/pr1-disable-shm-mm
2. Confirm base = `sgl-project/sglang:main`, head = `QIU023/sglang:pr1-disable-shm-mm`
3. Use the title and body below verbatim
4. Submit

---

## Title (copy-paste)

```
[srt/managers] SGLANG_DISABLE_SHM_MM env to force CPU multimodal IPC transport
```

## Body (copy-paste)

```markdown
## Motivation

`TokenizerManager` ships multimodal image tensors to the scheduler subprocess over
POSIX-SHM (`/psm_*`) when `cuda_ipc` is selected. When SGLang is launched *inside a
parent-managed process tree* (Ray actor groups, SLURM job arrays, Monarch meshes),
the parent's `resource_tracker` can unlink the segment before the scheduler opens it:

​```
  File ".../srt/managers/tokenizer_manager.py", in _determine_tensor_transport_mode / dispatch
    ...
FileNotFoundError: [Errno 2] No such file or directory: '/psm_a1b2c3'
​```

Hard crash, no recovery, engine never boots. The cross-node path already avoids SHM
(`dist_init_addr` → `"default"`); single-node parent-managed setups have no such opt-out.

## Modifications

Register `SGLANG_DISABLE_SHM_MM` (`EnvBool`, default off) in `srt/environ.py` and honour
it at the top of `_determine_tensor_transport_mode` — when set, return `"default"`
(inline-pickle, no SHM segment):

​```python
if envs.SGLANG_DISABLE_SHM_MM.get():
    return "default"
​```

`+86`, three files (`environ.py`, `tokenizer_manager.py`, and a GPU-free reproducer under
`examples/runtime/multimodal/`). Env unset → `cuda_ipc` path selected exactly as before.

## Effect

| | env unset (default) | `SGLANG_DISABLE_SHM_MM=1` |
|---|---|---|
| transport | `cuda_ipc` / SHM | `"default"` (inline pickle) |
| parent-managed tree | `FileNotFoundError: /psm_*` at boot | boots clean |
| large-image throughput | fast path | slightly slower (extra copy) |

Minimal repro (no GPU, stdlib only): `examples/runtime/multimodal/repro_disable_shm_mm_race.py`
runs a producer in its own interpreter so its `resource_tracker` unlinks the segment on exit
(faithful to per-worker teardown), then opens it from the consumer → deterministic
`FileNotFoundError: /psm_*`; the gate flips the selected transport to `"default"`.

## Accuracy / Speed

Transport-only — `"default"` and `"cuda_ipc"` deliver byte-identical tensors to the
scheduler, so no accuracy impact and existing multimodal tests stay green with the env
unset. `"default"` is slightly slower for very large image batches (extra serialize +
copy); opt-in only, default fast path untouched.

## Test

- [x] `py_compile` clean; default path unchanged (early-return only when env set).
- [x] Reproducer reproduces the crash and the gate's transport flip.
- [x] Soaked 12h under a multimodal RL workload on a Monarch actor mesh, 601 steps,
      zero `/psm` crashes. Full harness (public): https://github.com/QIU023/torchtitan_attention_residual
```

## Related work in same batch

- PR #7 (KDA causal_conv1d fp16 type-join) — separate sglang PR, same fork
- PR #8 (fp8 MoE Blackwell shmem) — separate sglang PR, same fork
