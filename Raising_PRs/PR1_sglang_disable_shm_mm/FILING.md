# PR #1 — filing instructions

## Status

🟢 **Branch pushed; PR not yet opened.**

| Item | Link / value |
|---|---|
| Fork branch | https://github.com/QIU023/sglang/tree/pr1-disable-shm-mm |
| Open-PR URL | https://github.com/QIU023/sglang/pull/new/pr1-disable-shm-mm |
| Target repo | https://github.com/sgl-project/sglang |
| Base | `sgl-project/sglang:main` |
| Head | `QIU023/sglang:pr1-disable-shm-mm` |
| Commit | `6d3718439` (1 commit, +9/-0 in `tokenizer_manager.py`) |
| Cross-link | (none — independent of other PRs) |

## To open the PR

1. Open https://github.com/QIU023/sglang/pull/new/pr1-disable-shm-mm
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
## Summary

Adds a `SGLANG_DISABLE_SHM_MM=1` env-gate at the top of
`TokenizerManager._determine_tensor_transport_mode` that forces the
multimodal payload IPC channel to use the `"default"` (pickled-tensor)
transport instead of `"cuda_ipc"` / SHM. The default behaviour is
unchanged when the env var is unset.

## Why

SHM-backed multimodal IPC races against Monarch / Ray / SLURM actor
lifecycles: when the spawning process's `resource_tracker` unlinks
`/psm_xxx` before SGLang's scheduler subprocess opens it, the engine
crashes at startup with:

```
FileNotFoundError: [Errno 2] No such file or directory: '/psm_xxx'
```

The trigger is timing-dependent and not always reproducible during
local single-process debugging, but is reliable under any
container-orchestrated multimodal workload (Monarch RL actor meshes,
Ray Serve deployments with multimodal models, SLURM job arrays).

The `"default"` transport inlines tensor bytes in the pickle — slower
for very large image batches but lifecycle-safe.

## Patch

Single hunk in `python/sglang/srt/managers/tokenizer_manager.py`,
inside `_determine_tensor_transport_mode`:

```python
def _determine_tensor_transport_mode(server_args: ServerArgs) -> TensorTransportMode:
    # SGLANG_DISABLE_SHM_MM=1: force "default" CPU transport instead of
    # "cuda_ipc"/SHM-backed multimodal payload IPC. SHM IPC races against
    # Monarch's actor lifecycle (the spawned actor's resource_tracker
    # unlinks /psm_xxx before SGLang's scheduler subprocess can open it).
    # "default" inlines the tensor in the pickle — slower for large image
    # batches but lifecycle-safe.
    import os as _os
    if _os.environ.get("SGLANG_DISABLE_SHM_MM", "0") == "1":
        return "default"
    is_cross_node = server_args.dist_init_addr
    ...
```

## Test plan

- [x] Static: `py_compile` on patched file (passes on Python 3.12).
- [x] Default behaviour unchanged: env unset → no code path change in
      the function below the early return.
- [ ] Functional smoke: launch SGLang Engine multimodal model under
      `SGLANG_DISABLE_SHM_MM=1`, verify it boots and serves an image
      generation request without `FileNotFoundError`. (We have run
      this end-to-end in our fork under Monarch RL actor mesh; happy
      to provide a containerised reproducer on request.)

## Backwards compatibility

100% — the patch is gated entirely on the env var, which defaults to
unset / `"0"`. No existing user is affected.
```

## Related work in same batch

- PR #7 (KDA causal_conv1d fp16 type-join) — separate sglang PR, same fork
- PR #8 (fp8 MoE Blackwell shmem) — separate sglang PR, same fork
