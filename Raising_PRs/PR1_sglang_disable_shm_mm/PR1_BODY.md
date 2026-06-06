## Motivation

`TokenizerManager` ships multimodal image tensors to the scheduler subprocess over
POSIX-SHM (`/psm_*`) when `cuda_ipc` is selected. When SGLang is launched *inside a
parent-managed process tree* (Ray actor groups, SLURM job arrays, Monarch meshes),
the parent's `resource_tracker` can unlink the segment before the scheduler opens it:

```
  File ".../sglang/srt/managers/mm_utils.py", line 1564, in __setstate__
    self._shm_handle = shared_memory.SharedMemory(name=self.shm_name)
  File ".../multiprocessing/shared_memory.py", line 104, in __init__
    self._fd = _posixshmem.shm_open(self._name, self._flags, mode=self._mode)
FileNotFoundError: [Errno 2] No such file or directory: '/psm_33bfb96e'
```

Hard crash, no recovery, engine never boots. The cross-node path already avoids SHM
(`dist_init_addr` → `"default"`); single-node parent-managed setups have no such opt-out.
This PR adds an opt-in env var that extends the same opt-out to them. Default behaviour is
unchanged when the env is not set.

## Modifications

Register `SGLANG_DISABLE_SHM_MM` (`EnvBool`, default off) in `srt/environ.py` and honour
it at the top of `_determine_tensor_transport_mode` — when set, return `"default"`
(inline-pickle, no SHM segment):

```python
if envs.SGLANG_DISABLE_SHM_MM.get():
    return "default"
```

`+4`, two files (`environ.py`, `tokenizer_manager.py`). Env unset → `cuda_ipc` path
selected exactly as before.

| | env unset (default) | `SGLANG_DISABLE_SHM_MM=1` |
|---|---|---|
| transport | `cuda_ipc` / SHM | `"default"` (inline pickle) |
| parent-managed tree | `FileNotFoundError: /psm_*` at boot | boots clean |
| large-image throughput | fast path | slightly slower (extra copy) |

Validation — real failure + fix, captured in our public research repo:

- **Before** (default `cuda_ipc`/SHM): a VLM GRPO rollout driven by an SGLang generator hit
  exactly the traceback above at a rollout boundary —
  [`grpo_overnight_v15_.../run.log` L235–241](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase11_rlhf_grpo_infra/rlhf/outputs/grpo_overnight_v15_20260512-184607/run.log#L235-L241).
- **After** (`SGLANG_DISABLE_SHM_MM=1`): a full **1200/1200-step** VLM GRPO soak (447M Kimi
  Block AttnRes, SGLang TP=4 generator, ~4h21m) completed with **zero `/psm` crashes** —
  run summary (env line + headline numbers):
  [`grpo_llava_kimi_overnight/SUMMARY.md`](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase11_rlhf_grpo_infra/rlhf/outputs/grpo_llava_kimi_overnight/SUMMARY.md).
- **Deterministic minimal repro** (no GPU, stdlib only):
  [`repro_disable_shm_mm_race.py`](https://github.com/QIU023/torchtitan_attention_residual/blob/main/Raising_PRs/PR1_sglang_disable_shm_mm/repro_disable_shm_mm_race.py)
  runs three back-to-back demos in one invocation, proving the fix end-to-end:

  1. **`demo_shm_race`** — producer in its own interpreter lets its
     `resource_tracker` unlink the segment on exit (faithful to per-worker
     teardown), consumer opens it → `FileNotFoundError: /psm_*` (the production crash).
  2. **`demo_default_no_race`** — the `"default"` transport that
     `SGLANG_DISABLE_SHM_MM=1` selects: payload travels as pickled bytes,
     **no `/psm_*` segment is ever created**, so the race is structurally impossible.
  3. **`demo_gate`** — patched env-gate decision: unset → `"cuda_ipc"`, `=1` → `"default"`.

  Sample output (Linux; deterministic across 3 runs, each segment name differs):

  ```text
  [default/cuda_ipc] CRASH as in prod: FileNotFoundError: /psm_a77f02a3
  [default] received 22528-byte payload via inline pickle (no /psm_* segment created -> race impossible)
  [gate] SGLANG_DISABLE_SHM_MM unset -> transport 'cuda_ipc'
  [gate] SGLANG_DISABLE_SHM_MM =1    -> transport 'default'
  ```

## Accuracy Tests

Not applicable — transport-only change. `"default"` and `"cuda_ipc"` deliver byte-identical
tensors to the scheduler (only the carrier differs), so model outputs are unaffected and the
existing multimodal tests stay green with the env unset.

## Speed Tests and Profiling

Opt-in only; the default `cuda_ipc` fast path is untouched, so there is no regression for
existing users. When `SGLANG_DISABLE_SHM_MM=1`, `"default"` inlines tensor bytes in the
pickle instead of passing a shared-memory handle — slightly slower for very large image
batches (extra serialize + copy). This is the intended lifecycle-safety trade-off.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). *(Verified locally against sglang's pinned hook versions on the 2 patched files: `isort 7.0.0`, `black 26.1.0`, `ruff 0.15.1` with sglang's `--select=F401,F821`, `codespell 2.4.1`, `trailing-whitespace`, `end-of-file-fixer`, `check-ast` — all clean.)*
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). *(No CI-registered test in this PR — the IPC lifecycle race needs a multi-process GPU harness. A self-contained, GPU-free reproducer is linked in Modifications; happy to add a registered test if maintainers prefer.)*
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). *(N/A — transport-only; rationale above.)*
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
