# PP Pressure Test — Blocked on Trunk Drift (2026-05-11)

## TL;DR

The AttnRes PP carrier (existing phase3 work) has bit-rotted against
recent torchtitan/torch.distributed.pipelining trunk. Three trunk-
drift bugs found in this debug session; first two fixable, third one
is a deeper API mismatch that requires non-trivial rework. Pressure
test sweep cannot run until that's fixed.

## What we exercised tonight

User wanted aggressive PP × VP smoke (PP=8 × VP=4 etc.) on deeper
synthetic carriers. Plan v3 design was sound (Path A, 32-layer +
48-layer Llama AttnRes carriers, naive vs adapter, from-scratch
1000-step C4 training).

Each of three sweep configurations hit a different layer of the
trunk-drift stack:

## Trunk-drift bugs

### Bug 1 (fixed) — `_set_lru_cache` missing on torch 2.9

`torchtitan/distributed/activation_checkpoint.py:232` unconditionally
calls `torch._C._dynamo.eval_frame._set_lru_cache(False)`. That
attribute only exists in torch 2.10 nightly. Workaround added: gate
the call with `hasattr`. Patched (uncommitted) in our submodule.

### Bug 2 (fixed) — `AttnResModel.forward(return_outputs=)` not accepted

Recent torchtitan trainer passes `return_outputs=False` through
`pp_schedule.step(**extra_kwargs)` → into model.forward. Our
AttnResModel didn't accept this kwarg. Workaround: added kwarg to
signature, ignore it (since AttnRes doesn't use SAC memory-budget
instrumentation). Patched (uncommitted).

### Bug 3 (BLOCKER) — Pipeline schedule deadlocks on multi-tensor stage I/O

Even at the simplest config (PP=8 VP=2 GBS=4 LBS=4 = 1 microbatch,
matching the 2026-04-20 successful pp8_adapter recipe.json exactly),
ranks deadlock in `batch_isend_irecv` at
`torch.distributed.pipelining.schedules._step_microbatches:1730`
with a 600s NCCL collective timeout.

Symptom: 5 GPUs at 100% util, 3 at 0% — classic "some ranks doing
forward, others stuck waiting for collective that never enqueued".

Root cause (hypothesis): AttnRes's non-last stage returns a
**2-tensor tuple** `(partial_block, stacked_blocks)`. Older
torch.distributed.pipelining inferred shapes from a single-tensor
trace; current trunk expects stages to declare their multi-tensor
I/O explicitly via `PipelineStage(input_args=..., output_args=...)`
or similar. Our `pipeline_adapter.py:pipeline_llm_with_cache_adapter`
doesn't pass these declarations through, so the schedule's send/recv
shape inference mismatches across ranks, deadlocks at PP P2P boundary.

The same deadlock affects the **naive path too** (pipeline_llm without
adapter), so it's not the adapter — it's the underlying carrier's
multi-tensor stage being mishandled.

This deadlock was NOT present in the 2026-04-20 phase3 runs that the
`runs/pp8_adapter` and `runs/pp8_naive` directories contain. Between
then and now, either torch or torchtitan's pipelining_fn API changed,
breaking the carrier.

## To unblock the pressure test

Need to (in this order):
1. Inspect torch.distributed.pipelining trunk for what changed in the
   multi-tensor stage I/O contract since April 2026.
2. Extend
   `torchtitan/experiments/attn_res/pipeline_adapter.py:pipeline_llm_with_cache_adapter`
   to declare each stage's input/output shape explicitly (likely by
   wiring the new `PipelineStage` constructor kwargs).
3. Add a smoke test in phase3 that runs PP=2 VP=1 GBS=2 LBS=1 (smallest
   PP config) and validates `step:1 loss:...` actually prints — this
   would have caught the trunk drift in CI.

Estimated effort: 4-8h. Out of scope for tonight; user wants results
sooner.

## What we DID validate tonight (worth keeping)

* L32_n8 and L48_n8 carrier registrations work — `model_registry`
  returns valid `ModelSpec`, no shape errors.
* Bugs 1 and 2 are real and now fixed in-tree (uncommitted; should
  commit if we want to land them upstream regardless of bug 3).
* `phase3/run_pp_pressure_test.sh` sweep harness works end-to-end
  on the bash side (spawns torchrun, captures train.log, builds
  SUMMARY.md) — once bug 3 is fixed the sweep will produce results
  without further changes.

## Recommendation

For now: punt on PP pressure test. Use the documented existing
2026-04-20 phase3 results (`runs/pp4_adapter_4gpu`, `runs/pp8_adapter`)
as the canonical adapter validation evidence. They proved naive-vs-
adapter loss match + reduced send-bytes at the small carrier scale.

The deeper-carrier sweep (L32, L48) is the right next step *once*
bug 3 is resolved — the new flavors are registered and the launcher
is ready.
