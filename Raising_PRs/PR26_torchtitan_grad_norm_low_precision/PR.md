# PR #26 — `clip_grad_norm_` total norm computed in the gradients' dtype (filed as upstream PR 4135)

**Status**: FILED as upstream PR 4135. Maintainer feedback on the first summary: "sorry I couldn't really understand the PR summary which seems to be written by AI." Body below is the rewrite per the CLAUDE.md PR-text rule -- Summary/Fix at 2-3 sentences, evidence as a runnable command plus one table, the 394-tensor split demo held back as follow-up ammo. Branch `grad-norm-fp32` (commit `5e88ff897`), evidence measured on upstream `f4e78188e`. 2026-08-14: reply + simplified description posted; Tianyu's follow-up confirms the root-cause reading ("get_total_norm doesn't support configurable norm dtype, or default to the safer fp32. Is that correct?") -- answer in the second reply block below. 2026-08-17: janeyx99 (pytorch core) invited a pytorch/pytorch issue and asked two design questions; both are answered in `ISSUE_pytorch.md`, which is now the live document for this PR. Nothing posted yet.
**Target**: `pytorch/torchtitan`, `torchtitan/distributed/utils.py` (`clip_grad_norm_` and `_clip_grad_norm_with_ep`)
**Risk**: no behaviour change under the default `training.dtype="float32"`; with `bfloat16` the reported and applied norm changes, by design.

---

## Reply to the review question ("what ops that were in bf16 are now in fp32?")

--- PASTE BEGIN ---

The computation is structurally identical before and after — per-tensor norms, then each group's norm-of-norms, then the cross-mesh combine; the fix only changes the dtype those two norm ops accumulate in (`get_total_norm` keeps the input dtype, so with bf16 grads they ran in bf16). Each group's partial norm is rounded before the combine in both cases — at bf16's 2^-8 step that rounding is large enough that different PP/EP layouts of the same gradients yield visibly different totals, at fp32's 2^-23 it is negligible, so all layouts agree. The gradients themselves stay bf16.

--- PASTE END ---

## Reply to the follow-up ("get_total_norm doesn't support configurable norm dtype, or default to the safer fp32. Is that correct?")

--- PASTE BEGIN ---

Yes, exactly. `get_total_norm` has no dtype parameter and inherits the input tensors' dtype — the ops underneath (`_foreach_norm` / `linalg.vector_norm`) both accept `dtype=`, it just never passes one. That's why the fix is a local mirror of `get_total_norm` with `dtype=torch.float32` on the per-tensor norm calls (the norm-of-norms then inherits fp32 from its inputs). Happy to propose a `dtype` argument on `get_total_norm` in pytorch/pytorch — defaulting to `None` to keep current behavior, since an unconditional fp32 would demote fp64 inputs — at which point this helper reduces to a one-line call.

--- PASTE END ---

## Replacement PR description

--- PASTE BEGIN ---

**Summary**

`get_total_norm` returns the norm in the input tensors' dtype, so with `training.dtype="bfloat16"` both call sites compute the per-tensor norms and the norm-of-norms in bf16 (3-4 significant digits). The rounding happens per group, so under PP/EP the total depends on how params are grouped across ranks: two runs with bit-identical gradients but different pipeline splits report different `grad_norm` and clip differently. The default `training.dtype="float32"` is unaffected.

**Evidence**

At `f4e78188e`, 4 GPUs, both dtypes, patch on/off:

```
torchrun --nproc_per_node=4 -m torchtitan.train --module llama3 --config llama3_debugmodel \
  --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 2 \
  --training.dtype bfloat16 --parallelism.data_parallel_shard_degree 2 \
  --parallelism.pipeline_parallel_degree 2 --training.local-batch-size 2
```

| `--training.dtype` | patch | step 1 | step 2 |
|---|---|---|---|
| float32 | before / after | 1.4509 | 1.6336 |
| bfloat16 | before | 1.4453 | 1.6328 |
| bfloat16 | after | **1.4508** | 1.6343 |

float32 is byte-identical before/after; patched bf16 lands next to float32 (1.4508 vs 1.4509).

**Fix**

Run those two norm ops with `dtype=torch.float32` via a local `_get_total_norm_fp32` that otherwise mirrors `get_total_norm`; gradients and the clip multiply are unchanged. It lives in this file rather than `torch.nn.utils` because `get_total_norm` has no `dtype` parameter and adding one is a PyTorch API change — happy to propose that upstream, at which point this helper reduces to a one-line call.

--- PASTE END ---

## Notes for the filer

- The one-sentence reply answers the maintainer's question first; post it before or with the description edit.
- Placement defense is pre-baked in the Fix paragraph: no `dtype` param on `get_total_norm` (PyTorch API change), and `distributed/utils.py` is already where torchtitan rewrites the torch.nn.utils clip path for DTensor/PP/EP. janeyx99 on the reviewer list is the likely source of that question.
- Held back for follow-up if asked: the 394-tensor split demo (same tensors, 121.000 / 121.153 / 121.051 / 121.012 vs fp32 exact 121.223) and the provenance story (two pipeline layouts, 788 bit-identical gradients, `grad_norm` 10.008054 vs 9.951641, true 9.989287).
- Do NOT frame as "bf16 training is broken": the gradients are fine, only the norm is mismeasured.
- The `pp_mesh` dtype fix (NCCL garbage on mismatched dtype) is an independent sibling patch in the same function; either can go first.

---

## Raw record — 2026-08-13, clean upstream worktree

Measured on `f4e78188e` after upstream regenerated its golden files (#4075 touched the c4_test loss path, #4099 the valid-token counts). Command as in the evidence block, run once per arm with the patch applied/reverted via `git checkout -- torchtitan/distributed/utils.py`, so the before-arms are upstream byte-for-byte. Earlier 681fd4b50 numbers (1.4485-era) are retired; a patched-bf16-equals-float32 reading from that run was a coincidence — bf16 inputs under an fp32 reduction approach the all-fp32 value rather than reach it.

Harness note: `--master_port` above 65536 fails with a rendezvous ValueError that looks like a patch failure and is not one.
