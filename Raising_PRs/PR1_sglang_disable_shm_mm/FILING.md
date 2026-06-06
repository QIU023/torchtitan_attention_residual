# PR #1 — filing instructions

## Status

🟢 **Force-pushed 2026-06-06 (`origin/pr1-disable-shm-mm` @ `8f861043e4`); PR not yet opened on GitHub.**

| Item | Link / value |
|---|---|
| Fork branch | https://github.com/QIU023/sglang/tree/pr1-disable-shm-mm |
| Open-PR URL | https://github.com/QIU023/sglang/pull/new/pr1-disable-shm-mm |
| Target repo | https://github.com/sgl-project/sglang |
| Base | `sgl-project/sglang:main` @ `4ef081b903` (2026-06-05) |
| Head | `QIU023/sglang:pr1-disable-shm-mm` |
| Commit | `8f861043e4` (1 commit, **2 files / +4** — `environ.py +1`, `tokenizer_manager.py +3`). Code identical to `f98e867c02`; subject describes both the action (`Add SGLANG_DISABLE_SHM_MM`) and the trigger (`when /psm_* races parent process-tree lifecycle`) without "opt-out" framing; body links the strengthened reproducer. |
| Prior HEADs | `4bcf838df` (raw-os) → `cbf984255d` (rebased) → `c381a13905` (idiomatic) → `3f4e2a3fb4` (repro in examples) → `464687ebe3` (repro removed) → `f98e867c02` (trimmed verbose comments) → `79ce6540c1` (body links 3-demo reproducer) → `4009857ba5` (Fix-first phrasing) → `8f861043e4` (Add + trigger-clause phrasing) |
| Cross-link | (none — independent of other PRs) |

> **Re-prepared 2026-06-05.** Three changes vs. the old single-hunk version:
> 1. **Re-rebased** onto current `upstream/main` (`4ef081b903`, +308 commits past the
>    2026-05-21 base); `_determine_tensor_transport_mode` byte-identical upstream → clean replay.
> 2. **Made idiomatic** per the repo's `env-var-conventions` skill: registered
>    `SGLANG_DISABLE_SHM_MM = EnvBool(False)` in `srt/environ.py` (VLM Item CUDA IPC
>    Transport group) and read it via `envs.SGLANG_DISABLE_SHM_MM.get()` instead of a
>    raw `os.environ.get` — heads off the predictable "use environ.py" review bounce.
> 3. **Wrote a GPU-free, stdlib-only reproducer** — but kept it in **this** repo at
>    `Raising_PRs/PR1_sglang_disable_shm_mm/repro_disable_shm_mm_race.py`, **not** in
>    sglang's `examples/` (that tree is for first-class serving examples, not ad-hoc
>    repros). The PR body links it. The producer runs as its own interpreter so *its*
>    resource_tracker unlinks the segment on exit — deterministic, faithful to the
>    Ray/SLURM/Monarch per-worker teardown.
>
> py_compile clean on both PR files. **Reproducer strengthened 2026-06-06** with a third
> demo (`demo_default_no_race`) that runs the gated `"default"` path end-to-end and
> proves no `/psm_*` segment is created — making the race structurally impossible on the
> patched path. **Verified on WSL2 / Linux: 3/3 runs**, crash + no-crash + gate all
> deterministic (Windows shows the same `/psm_*` race under the `wnsm_` backend).
> **Force-pushed to `origin/pr1-disable-shm-mm` (`f98e867c02`).** The PR diff is
> now **only the 2 source files** (`environ.py +1`, `tokenizer_manager.py +3`) — verbose
> comments trimmed to a single line matching the existing `is_cross_node` style. Remaining step:
> open the PR on github.com/sgl-project/sglang (manual web — `gh` not installed here).

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

1. Open https://github.com/QIU023/sglang/pull/new/pr1-disable-shm-mm (branch already pushed)
2. Confirm base = `sgl-project/sglang:main`, head = `QIU023/sglang:pr1-disable-shm-mm`
3. Use the title and body below verbatim
4. Submit

---

## Title (copy-paste)

```
[srt/managers] Add SGLANG_DISABLE_SHM_MM to fall back to default transport when /psm_* races parent process-tree lifecycle
```

## Body (copy-paste)

The PR description is maintained as a **standalone file**, aligned to the sglang official
PR template (Motivation / Modifications / Accuracy Tests / Speed Tests and Profiling /
Checklist / Review and Merge Process) with clean code fences:

→ [`PR1_BODY.md`](./PR1_BODY.md) — open it, select-all, paste into the PR description box.

(Kept out of this file on purpose: embedding it here forced zero-width-space hacks on the
nested code fences and drifted out of sync. Single source of truth = `PR1_BODY.md`.)

## Reproducer

`repro_disable_shm_mm_race.py` (this folder) — GPU-free, stdlib-only, deterministic
`/psm_*` crash + gate flip. **Not** shipped in the sglang PR (kept out of upstream
`examples/`); the PR body links it here in the public repo.

## Related work in same batch

- PR #7 (KDA causal_conv1d fp16 type-join) — separate sglang PR, same fork
- PR #8 (fp8 MoE Blackwell shmem) — separate sglang PR, same fork
