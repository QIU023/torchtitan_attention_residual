# PR #26 — `clip_grad_norm_`: the total norm is computed in the gradients' dtype, so with `training.dtype=bfloat16` the clipped update depends on how the pipeline was cut

**Status**: ✅ ready to file — reproduced on a CLEAN upstream worktree (`681fd4b50`) with the unmodified `llama3_debugmodel` and no fork code in the loop. Re-audited 2026-08-13 against upstream `f4e78188e`: all three `get_total_norm` call sites unchanged, patch applies cleanly, `Iterable`/`DTensor` imports present, `_foreach_norm`/`vector_norm` dtype args verified. Branch `grad-norm-fp32` pushed (commit `7c98c6c51`, see commits.md).
**Target**: `pytorch/torchtitan`, `torchtitan/distributed/utils.py` (`clip_grad_norm_` and `_clip_grad_norm_with_ep`)
**Risk**: low in shape, visible in numbers. No behaviour change when gradients are float32, which is the default; with `training.dtype="bfloat16"` the reported and applied norm changes, by design — the old value was inaccurate.

**Format note**: single-line paragraphs (tables, lists and code blocks excepted) so the body copies verbatim.

---

## Suggested PR title

> clip_grad_norm_: compute the total norm in float32 so it does not depend on the parallelism partition

## Suggested PR body

--- PASTE BEGIN ---

### Summary

`torch.nn.utils.get_total_norm` returns the norm in the **input tensors' dtype**. Both call sites in `torchtitan/distributed/utils.py` pass gradients straight in, so when gradients are bfloat16 the per-tensor norms and the norm-of-norms are both bfloat16 — three to four significant digits.

Two consequences:

1. The reported `grad_norm` is wrong by a few tenths of a percent.
2. The error depends on **how the tensors are grouped**. Under PP each rank norms its own share and the results are combined across `pp_mesh`; under EP the parameters are split into an EP group and a non-EP group and combined. So the value depends on where the pipeline was cut, not only on the gradients.

Point 2 is the one that matters, because `max_norm` is usually small enough that clipping fires every step. The clip factor is `max_norm / total_norm`, so an 0.2% error in the norm is an 0.2% error in that step's effective learning rate — and two runs with identical gradients but different PP splits take different-sized steps.

Gradients are float32 by default (`training.dtype="float32"` gives fp32 master weights), so this is invisible in the common configuration. It appears with `training.dtype="bfloat16"`, which is a documented option in `configs.py` (`dtype: Literal["bfloat16", "float32"] = "float32"`).

### Evidence — unmodified `llama3_debugmodel` on a clean checkout

`681fd4b50`, no fork code. 4 GPUs, `dp_shard=2 x pp=2`, seed 42, deterministic, 2 steps, `--training.local-batch-size 2`:

| `--training.dtype` | patch | step 1 `grad_norm` | step 2 `grad_norm` |
|---|---|---|---|
| `float32` | before | 1.4485 | 1.6350 |
| `float32` | after | 1.4485 | 1.6350 |
| `bfloat16` | before | 1.4453 | 1.6328 |
| `bfloat16` | **after** | **1.4485** | 1.6357 |

Two things to read off it. float32 is untouched, as it must be — there is nothing to fix when the gradients are already float32. And the patched bfloat16 run reports the same step-1 norm as the float32 run, 1.4485, while unpatched bfloat16 reports 1.4453: computing the norm in float32 recovers the value the higher-precision configuration agrees on.

### Evidence — the partition dependence, without any distributed setup

394 bfloat16 tensors with magnitudes spanning 1e-4 to 1e-1, norms taken with the same function the two call sites use, then the same tensors split into two groups and combined the way `pp_mesh`/EP combines them:

```text
float32 exact          121.222923
get_total_norm         121.000000     0.184% error, dtype=torch.bfloat16
  split 100 / 294      121.153351     0.057%
  split 200 / 194      121.050613     0.142%
  split 300 / 94       121.011543     0.174%
```

Same gradients, three different answers. The returned dtype is `torch.bfloat16`, which is the whole mechanism: `121.000000` is what a bf16 scalar can represent near 121.

### Fix

Compute the norm in float32. `get_total_norm` takes no dtype argument, so the patch adds a local `_get_total_norm_fp32` that mirrors it — same empty-list result, same `error_if_nonfinite` behaviour, same foreach fast path where it applies — and passes `dtype=torch.float32` to the norm calls:

* plain tensors: `torch._foreach_norm(tensors, norm_type, dtype=torch.float32)` (`foreach=False` is honored with a per-tensor fallback)
* DTensors: `torch.linalg.vector_norm(t, norm_type, dtype=torch.float32)` one at a time, because upstream's foreach support check excludes them as well. The dtype argument preserves the `_NormPartial` placement, so the existing `full_tensor()` reduction in both call sites still works unchanged.

Three call sites redirected (one in `clip_grad_norm_`, two in `_clip_grad_norm_with_ep`). Diff is +55/−3 in one file.

An alternative worth mentioning: add a `dtype` argument to `torch.nn.utils.get_total_norm` in PyTorch and drop this helper. That is the better long-term home, and this patch is compatible with it — the helper becomes a one-line call.

### Test plan

* `llama3_debugmodel`, `dp_shard=2 x pp=2`, `--training.dtype float32`: `grad_norm` unchanged (1.4485 / 1.6350 before and after).
* Same with `--training.dtype bfloat16`: `grad_norm` moves to the float32-accurate value (1.4453 -> 1.4485 at step 1).
* A unit test can pin the mechanism without a cluster: build a list of bf16 tensors, assert `get_total_norm` differs from the float32 computation and that splitting the list changes the combined result, then assert the new helper is split-invariant. Happy to add it during review; it needs no GPU.

--- PASTE END ---

## Notes for the filer

- Lead with the `llama3_debugmodel` table. It needs none of our model code, and the `float32` rows show the change is inert in the default configuration — that is the reviewer's first question.
- Second: the synthetic 394-tensor split table. It is the clearest statement of the actual defect (same gradients, three answers) and runs in one CPU process.
- Provenance, one sentence: found while aligning two pipeline layouts of the same multimodal model, where identical gradients produced `grad_norm` 10.008054 vs 9.951641 -- the true float32 value being 9.989287, so both reported numbers were wrong in opposite directions.
- Do NOT frame this as "bf16 training is broken". The gradients are fine; only the norm is mismeasured. Every per-parameter gradient in the case above was bit-identical between the two layouts.
- Expect the question "why not just recommend float32". Answer: `training.dtype="bfloat16"` is a supported option, and while it is selected the norm should still be right; also the partition dependence makes two otherwise-identical runs differ, which is a reproducibility problem independent of anyone's precision preference.
- Pair with the `pp_mesh` dtype fix in the same function if both are filed together (see commits.md): that one is a dtype mismatch NCCL turns into garbage, this one is a precision loss the partition makes visible. Same function, same family. They are independent patches and either can go first.

---

## Raw record — 2026-08-09, clean upstream worktree

Worktree at `681fd4b50` (`git worktree add --detach /tmp/tt_upstream 681fd4b509b1`), `ls torchtitan/models/` confirmed no `kimi_k3`:

```text
common deepseek_v3 flux gpt_oss kimi_k2_7 llama3 qwen3 qwen3_5
```

Command (both arms identical apart from `--training.dtype` and whether the patch was applied):

```text
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m torchtitan.train \
  --module llama3 --config llama3_debugmodel --debug.seed 42 --debug.deterministic \
  --metrics.log_freq 1 --training.steps 2 --training.dtype <float32|bfloat16> \
  --parallelism.data_parallel_shard_degree 2 --parallelism.pipeline_parallel_degree 2 \
  --training.local-batch-size 2 --dump-folder /tmp/up
```

Results as tabulated above. The patch was applied to and reverted from the worktree with `git checkout -- torchtitan/distributed/utils.py` between arms, so the before-arms are upstream byte-for-byte.

One harness note for whoever reproduces: `--master_port` above 65536 fails with `ValueError: The port number of the rendezvous endpoint ... must be an integer between 0 and 65536`, which looks like a patch failure and is not one.
