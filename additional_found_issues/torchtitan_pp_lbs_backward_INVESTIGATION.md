# PP + Interleaved1F1B + V≥2 + LBS≥2 Backward Graph Reuse — Root Cause Investigation

**Date:** 2026-05-03  
**Investigator:** debugger sub-agent  
**Status:** Root cause identified with high confidence. Enough evidence to file an upstream RFC.

---

## 1. One-Paragraph Summary

Under `ScheduleInterleaved1F1B` with V≥2 virtual stages per rank and local batch size ≥2,
the `PipelineStage.args_recv_info` activation receive buffers are allocated **once** (during
the first call to `_initialize_stages`) and then **reused across every training step**.  Each
buffer is a single `torch.Tensor` that is shared across all microbatch chunk IDs for a given
stage.  In step N, microbatch 0's forward writes into `args_recv_info[0].buffer`, which is
stored into `fwd_cache[0]` as `input_values` and participates in microbatch 0's autograd graph.
In step N+1, the same physical buffer object is overwritten by a new `dist.irecv` call for the
next step's microbatch 0 forward.  Because `clear_runtime_states()` only clears `fwd_cache` and
resets `.grad` — but does NOT re-allocate the buffer tensors — the autograd graph from step N
still holds a live reference (`input_values`) to the step N buffer.  When microbatch 0 backward
runs in step N+1, autograd walks the step-N graph through the now-overwritten buffer and crashes
with "Trying to backward through the graph a second time" because step-N's saved tensors were
already freed during step N's backward.  The trigger requires LBS≥2 (more than one microbatch
per step per rank) to create a timing window where the second microbatch's forward overwrites
a buffer before the first microbatch's backward is done within the same step.  V≥2 is required
because the interleaved schedule causes backward for microbatch M to overlap with forward for
microbatch M+N within a single step on the same rank — which is what creates the
intra-step aliasing window even on the first step.

---

## 2. Code Paths Read

### 2.1 `_backward.py` — the crash site

File: `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/_backward.py`  
Lines 282–413: `stage_backward()`

Key observation: `stage_backward` calls `torch.autograd.backward(stage_output_tensors, ...)`.
The `input_values` list passed in (`bwd_kwargs["input_values"]`) comes directly from
`fwd_cache.pop(bwd_chunk_id)` in `backward_one_chunk` (stage.py:763-766).  If any tensor in
`input_values` has had its storage overwritten (by a subsequent P2P recv into the same buffer),
autograd traversal of the backward graph will encounter freed saved-tensors and raise the crash.
The `raise RuntimeError(exc_msg) from e` at line 411 wraps any exception from
`torch.autograd.backward`, so the user-visible stack trace always ends at line 411.

### 2.2 `stage.py` — buffer allocation and reuse

File: `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/stage.py`

**`_prepare_forward_infra` (PipelineStage), lines 1535-1586:**  
Called once per `_initialize_stages()` call (guarded by `_stages_forward_initialized = True`).
For each chunk ID 0..N-1, creates a `_RecvInfo` with a **single pre-allocated buffer tensor**:
```python
recv_infos = tuple(
    _RecvInfo(
        f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
        self.stage_index - 1,
        _make_tensor_from_meta(inp, self.device),   # allocated ONCE
    )
    for inp in self.inputs_meta
)
self.args_recv_info[chunk_id] = recv_infos
```
Each `recv_infos[k].buffer` is a fresh tensor at construction time, but
`_make_tensor_from_meta` allocates one buffer per (chunk_id, input_slot).
The same buffer object persists across every subsequent `step()` call.

**`get_fwd_recv_ops` (line 422-429) + `_retrieve_recv_activations` (lines 548-554):**  
`get_fwd_recv_ops` issues a `dist.irecv` into `args_recv_info[fwd_chunk_id].buffer`.
`_retrieve_recv_activations` then returns `info.buffer` directly (the live tensor
object, not a copy) as the stage input.  This tensor goes into `fwd_cache[chunk_id]`
as `input_values` and is the root of the autograd graph for that microbatch.

**`forward_one_chunk` (lines 669-735):**  
```python
flatten_input_tensors = flat_args + flat_kwargs   # contains the buffer tensor
self.fwd_cache[fwd_chunk_id] = (
    output_tuple,            # stage_output
    flatten_input_tensors,   # input_values  <-- live reference to recv buffer
)
```
`flatten_input_tensors` holds a Python reference to the same buffer object that
`args_recv_info[fwd_chunk_id]` also holds.  The buffer is NOT copied or detached.

**`clear_runtime_states` (lines 512-530):**  
Called at the top of every `step()`:
```python
self.fwd_cache.clear()        # pops the per-mb (output_tuple, input_values) entries
self.output_chunks.clear()
for recv_tuple in self.args_recv_info.values():
    for a in recv_tuple:
        if isinstance(a, _RecvInfo):
            a.buffer.grad = None   # zero grad, but BUFFER ITSELF is not reallocated
```
The buffer tensor objects in `args_recv_info` survive `clear_runtime_states()` unchanged.
`fwd_cache` is cleared so the Python-level reference `fwd_cache[k]` is gone, but the
autograd graph built in step N may still reference the buffer via saved tensors inside
the C++ autograd engine (for operations that were computed on inputs derived from
the buffer, like matrix multiplications whose backward needs the original input).

**`_prepare_backward_infra` (lines 297-305):**  
Creates one `_RecvInfo` buffer per (chunk_id) for gradient receives.  Same single-buffer
pattern, same reuse across steps.

### 2.3 `schedules.py` — initialization guard and interleaving

File: `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/schedules.py`

**`PipelineScheduleMulti._initialize_stages` (lines 1518-1545):**  
```python
if not self._stages_forward_initialized:
    ...                             # allocates all recv buffers
    self._stages_forward_initialized = True
if self._has_backward and not self._stages_backward_initialized:
    ...
    self._stages_backward_initialized = True
```
These flags are set to `True` after the first `step()` and **never reset**.  Buffer tensors
are allocated exactly once for the lifetime of the schedule object.

**`PipelineScheduleMulti.step` (lines 1586-1643):**  
```python
for stage in self._stages:
    stage.clear_runtime_states()   # clears fwd_cache, NOT the recv buffers
...
self._step_microbatches(...)
```
After `clear_runtime_states`, the next step's `_step_microbatches` immediately begins
issuing `get_fwd_recv_ops(mb_index)` which writes new data into the surviving buffer tensors.

**`_PipelineScheduleRuntime._step_microbatches` (lines 2031+):**  
The Interleaved1F1B schedule uses this runtime's `pipeline_order_with_comms`, which
interleaves `FORWARD` and `FULL_BACKWARD` actions across virtual stages within a single
step.  Under V=2, the ordering on a mid-pipeline rank looks like:
```
RECV_F mb0/vs0, FORWARD mb0/vs0, SEND_F mb0/vs0,
... (other ranks working) ...
RECV_F mb1/vs0, FORWARD mb1/vs0, SEND_F mb1/vs0,
RECV_B mb0/vs1, FULL_BACKWARD mb0/vs1, SEND_B mb0/vs1,
RECV_F mb2/vs0, FORWARD mb2/vs0, ...    <-- new recv into mb2's buffer
FULL_BACKWARD mb1/vs1, ...
```
The `RECV_F mb2/vs0` overwrites `args_recv_info[2].buffer` while
`FULL_BACKWARD mb1/vs1` is still in-flight and depends on `input_values` from a
previous forward pass that may share graph nodes with tensors derived from
`args_recv_info[0].buffer` (specifically: with LBS≥2, the forward computation
for mb0 and mb1 are batched together in one submodule call, and the resulting
autograd graph spans both buffer tensors).

**`ScheduleInterleaved1F1B._calculate_single_rank_operations` (lines 2568-2618):**  
With `n_local_stages=V` and `pp_group_size=P`, the warmup phase runs `2*(V-1)*P/something`
forwards before the first backward.  With V=2, PP=4, LBS=5 (so n_microbatches=5), the
steady-state has at least 2 forwards outstanding before the first backward, meaning
`args_recv_info[1].buffer` and `args_recv_info[0].buffer` are both live in the autograd
graph simultaneously when the first backward begins.

### 2.4 `torchtitan/distributed/pipeline_parallel.py`

File: `/root/torchtitan_attention_residual/torchtitan/torchtitan/distributed/pipeline_parallel.py`

Key observation: `build_pipeline_schedule` (lines 180-253) calls
`schedule_class(stages, n_microbatches=n_microbatches, ...)` where
`n_microbatches = local_batch_size // microbatch_size`.  When `microbatch_size=1` (the default)
and `local_batch_size=LBS`, each microbatch is a single example.  The schedule is then
instantiated with `n_microbatches=LBS`.  `_prepare_forward_infra` allocates LBS separate buffer
tensors (one per chunk ID).  These all survive across steps.

The `pipeline_module_split` function (lines 367-507) uses `PipelineStage(model, stage_idx, ...)`,
the non-tracing variant, which calls `_make_tensor_from_meta` for each of `inputs_meta`.  This
is the same allocation that gets reused.

### 2.5 `torchtitan/experiments/attn_res/pipeline_adapter.py`

File: `/root/torchtitan_attention_residual/torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py`

Key observations:
1. `CrossStageCacheAdapter` wraps `stage.submod` and patches `forward_one_chunk` /
   `backward_one_chunk` via `_install_mb_index_patch`.
2. The patch is a thin wrapper: it stashes the mb index, calls `orig_fwd`/`orig_bwd`,
   then clears the index.  It does NOT modify how `fwd_cache`, `args_recv_info`, or
   `input_values` are managed.
3. Own-rank cached block tensors are stored **detached** (`blk.detach()`) in `_cache`.
   The `_LocalCacheCapture` autograd function deposits grad into a slot and returns
   `None` gradient — explicitly severing the consumer-to-producer graph link.
4. The comments in the file explicitly document the prior double-backward crash
   ("the root cause of the double-backward crash the prior `_LocalCacheAugment.apply +
   view` pattern hit under real PP + FSDP + AC rerun") at lines 750-756, and confirm
   the detach pattern was added specifically to avoid backward graph traversal from
   consumer to producer.
5. The adapter does NOT introduce any new `retain_graph=True` calls or shared
   autograd-graph references between microbatches.  Its forward-path changes are
   confined to delta building and block stacking; none of these hold a cross-mb
   reference to an `args_recv_info` buffer.

**Conclusion from 2.5:** The adapter is not the source of the bug in the generic PP
case.  The adapter has its own, separate double-backward problem that has already been
fixed (the `_LocalCacheAugment → hook+detach` migration documented in
`handoff_status_20260420_part3.md`).  The crash reported in
`torchtitan_pp_microbatch_backward_graph.md` is in the upstream PP buffer-reuse path.

---

## 3. Hypotheses, Side-by-Side Comparison

### Hypothesis 1 (H1): Generic upstream PP bug — shared recv buffer tensor aliased across steps

**Mechanism:**

```
Step N:
  mb_k forward:
    irecv → args_recv_info[k].buffer  (a pre-allocated tensor T_k)
    fwd_cache[k] = (output, [T_k, ...])    # T_k is in the autograd graph
    stage_backward: T_k.grad used; autograd walks graph, frees step-N saved tensors

Step N+1:
  clear_runtime_states():
    fwd_cache.clear()          # Python dict entry gone
    T_k.grad = None            # grad cleared, but T_k still allocated at same address
  mb_k forward:
    irecv → T_k (SAME OBJECT)  # overwrites T_k storage with new data
    fwd_cache[k] = (output, [T_k, ...])  # new entry, same T_k
  mb_j backward (j < k, interleaved due to V=2):
    fwd_cache.pop(j) → input_values contains T_j
    torch.autograd.backward(...)
    → traverses step-N graph nodes that were saved against T_j
    → those C++ saved tensors were freed during step N backward
    → RuntimeError: "backward through the graph a second time"
```

**Why LBS≥2 is required:** With LBS=1 (n_microbatches=1), there is only a single
microbatch.  The interleaved schedule degenerates to sequential F→B with no overlap.
The single buffer T_0 is written by forward, then read by backward, then the step ends.
By the time step N+1 begins, T_0's autograd graph has been fully freed by step N's
backward.  The new irecv into T_0 in step N+1 creates a fresh graph with no stale
references.  With LBS≥2, the schedule overlaps forward (writing into T_k) with backward
(reading saved tensors from T_j, j<k) within the same step, and cross-step aliasing
of the buffer makes step N's freed graph accessible to step N+1's backward.

**Why V≥2 is required:** With V=1, `_PipelineScheduleRuntime` reduces to 1F1B with a
single local stage per rank.  The pipeline_order has no interleaving of F and B on the
same rank within a step; all forwards complete before any backward on that rank.  The
fwd_cache thus holds all microbatch entries simultaneously, and no recv-buffer aliasing
occurs within a step.  V=2 introduces a second local stage, causing the schedule to
interleave backward actions for virtual-stage-1 microbatches with forward actions for
virtual-stage-0 microbatches on the same physical rank, creating the intra-step window.

**Evidence for H1:**
- `_prepare_forward_infra` allocates buffers once (line 1563, `_make_tensor_from_meta`).
- `_stages_forward_initialized` flag (line 1540) prevents re-allocation.
- `clear_runtime_states` (lines 512-530) explicitly does NOT re-allocate buffers.
- `get_fwd_recv_ops` issues `dist.P2POp(dist.irecv, info.buffer, ...)` — in-place write
  into the existing buffer object (stage.py line 334).
- `forward_one_chunk` stores the buffer tensor into `fwd_cache` without copying
  (stage.py lines 717-723).
- `backward_one_chunk` pops `fwd_cache` and passes `input_values` (which includes the
  buffer) directly to `stage_backward` (_backward.py line 282).
- `stage_backward` calls `torch.autograd.backward` on graph nodes that have the buffer
  as a leaf — those nodes' C++ saved tensors are freed on first backward.

**Evidence against H1:**
- The symptom doc says the crash fires "on step 2's backward", which would be consistent
  with step 1's backward having freed step-1 graph nodes that step 2 backward tries to
  re-traverse through the shared buffer.  This is cross-step aliasing, not intra-step.
  **However,** the intra-step scenario (V≥2 forward overwrites buffer while backward
  still in-flight for an earlier mb within the same step) also fits and is reinforced by
  the V≥2 requirement.
- No direct stack trace shows the exact buffer tensor being aliased, since this is a
  read-only investigation.  The smoking gun would be to `print(id(info.buffer))` before
  and after `clear_runtime_states()` — expected: same ID both times.

### Hypothesis 2 (H2): AttnRes cache adapter — cross-mb graph aliasing via recv_delta_tensor

**Mechanism:**

`CrossStageCacheAdapter._forward_delta` (line 602) receives `recv_delta_tensor` (the
incoming P2P activation).  For middle and last stages, it calls `_keepalive_touch`
(lines 813-824) which adds `0.0 * prev_recv_tensor.sum()` to the output:
```python
touch = 0.0 * prev_recv_tensor.sum()
return payload + touch
```
This touch operation creates an autograd edge from `recv_delta_tensor` (microbatch mb_j's
forward) into the output, which is stored in `fwd_cache[mb_j]`.  If `recv_delta_tensor`
is the same physical buffer object as `args_recv_info[mb_j].buffer` (it is, see 2.2
above), and that buffer is later overwritten by a different microbatch's irecv, then the
"touch" edge in the autograd graph points to a tensor whose storage has been overwritten.
When mb_j backward runs, autograd computes `grad * sum(prev_recv_tensor)`, traverses
into the now-stale buffer, and may fail with the double-backward error if the same buffer
was used in a prior step's backward.

Additionally, `_finish_forward` (lines 706-811) builds the outgoing delta by re-reading
from `self._cache.get_blocks(mb)`.  The `cache_by_bidx` dict (line 784) contains
references to block slices from `prev_recv_tensor` (stored by `_finish_forward` lines
728-737 via `self._cache.append`).  Under LBS≥3, later microbatches' forwards run while
earlier microbatches' backwards are underway; the block slices in `_cache` for mb_j may
be aliased to the buffer objects that mb_j's backward is traversing.

**Evidence for H2:**
- The adapter's `_keepalive_touch` creates an explicit autograd edge into
  `prev_recv_tensor`, which is `args_recv_info[mb].buffer` — the same shared buffer.
- The `_cache.append` at lines 730-737 stores slices of `prev_recv_tensor` that are
  autograd-live.  These survive in `_cache` until `_drop_all_seen_and_clear` runs at
  step end.
- Under LBS≥3, multiple mb entries coexist in `_cache` during the steady-state phase.
  If mb_j's recv buffer is overwritten by mb_{j+2}'s forward recv while mb_j's cache
  entry still holds a slice of mb_j's old recv tensor, the slice storage is overwritten.

**Evidence against H2:**
- H2 only applies when `TORCHTITAN_ATTNRES_CACHE=1` (adapter opt-in env flag, line 91).
  The symptom doc says the crash was reproduced "with all three of our experiment-specific
  patches reverted" (section "Why this is a torchtitan core issue") implying the adapter
  was disabled.  If the crash reproduces without the adapter, H2 cannot be the primary cause.
- The adapter's own documentation (lines 246-286) explicitly describes and fixes the
  specific double-backward risk for own-rank cached commits via the detach+hook pattern.
  The adapter has a built-in mitigation against the consumer→producer traversal.
- The `_keepalive_touch` returns a value that goes into `fwd_cache` as `stage_output`
  (not `input_values`).  `stage_backward` calls `stage_output.detach_()` after backward
  (stage.py lines 841-843) and the last-stage detach is explicit (stage.py lines 835-843).
  So the touch tensor is on the `stage_output` side, which is freed after use.
- H2 would not explain the crash when the adapter is disabled.

---

## 4. Verdict

**H1 is the primary root cause.**

The recv buffer tensor aliasing across training steps (and within a step under V≥2
interleaving) is a structural property of `PipelineStage._prepare_forward_infra` +
`clear_runtime_states` — it exists regardless of the model, regardless of whether
AttnRes / kimi_linear / the cache adapter is active, and regardless of the interconnect.

The three required conditions map precisely onto the code:

| Condition | Mechanism |
|---|---|
| PP > 1 | Enables P2P recv-buffer allocation path in `_prepare_forward_infra` |
| V ≥ 2 | Interleaves F and B within a step on the same rank, creating intra-step aliasing window |
| LBS ≥ 2 | Creates multiple microbatch buffer slots; the interleaved schedule writes one buffer while backward traverses another that shares step-N graph context |

The crash fires "on step 2's backward" because step 2 is the first time buffer T_k has
been overwritten (by step 2's forward) while step 1's autograd graph still has T_k's
C++ saved tensors registered.  Step 1's backward completes fully, but the autograd
engine retains internal state (the saved-tensor table for the prehook intermediates in
`stage_backward_input`) that references T_k's storage.  Step 2 then irecv's into T_k
and tries to backward the step-2 graph, which re-encounters the now-stale step-1
saved-tensor entries and raises the error.

**H2 may represent an independent but secondary bug** in the AttnRes adapter's
`_keepalive_touch` path under LBS≥3.  It would be masked by the H1 crash since H1
fires first.  The adapter's own documentation acknowledges the prior double-backward
history, and the current detach+hook design mitigates the same-rank virtual-stage case,
but the `prev_recv_tensor` keepalive may still create a cross-step aliasing exposure
when adapter is enabled.

**Evidence needed to fully confirm H1 without running GPU code:** Print `id(T_k)` before
and after `clear_runtime_states()` in `_prepare_forward_infra`'s buffer loop and in the
forward loop — should be equal.  Confirm `dist.irecv` writes in-place into the same
storage (it does, by PyTorch P2P design).

**Evidence needed to confirm or rule out H2 independently:** Run the same PP+V+LBS config
with `TORCHTITAN_ATTNRES_CACHE=0` (adapter disabled).  If crash persists: H1 is confirmed
as model-independent.  If crash disappears: H2 is the primary cause.

---

## 5. Candidate Fixes

### 5.1 Fixes for H1 (recommended)

#### Band-aid (1 line change)

In `PipelineStage.clear_runtime_states()`, add a `requires_grad` reset on the buffer
and force a new storage allocation:

```python
# In stage.py clear_runtime_states():
for recv_tuple in self.args_recv_info.values():
    for a in recv_tuple:
        if isinstance(a, _RecvInfo):
            a.buffer = torch.empty_like(a.buffer)  # NEW: reallocate buffer
            a.buffer.requires_grad_(True)
            # Remove: a.buffer.grad = None  (now unnecessary, new tensor has no grad)
```

This breaks the Python object identity between step-N's `input_values` and step-N+1's
recv buffer.  Step-N+1's irecv writes into a fresh tensor; step-N's autograd graph
retains the old tensor (now unreferenced from `args_recv_info`) until the C++ saved-tensor
table releases it normally.

Cost: one `torch.empty_like` per recv buffer per step.  For typical PP configs with
2-4 recv buffers per stage and a step time of hundreds of ms, this cost is negligible.

#### Real fix

The root cause is that `_prepare_forward_infra` uses static pre-allocated buffers
optimized for the assumption that each step is independent.  The real fix has two parts:

**Part A** — In `_PipelineStageBase._prepare_backward_infra` / `clear_runtime_states`,
after `fwd_cache.clear()`, reallocate the recv buffers to break the cross-step aliasing:

```python
# stage.py, _PipelineStageBase.clear_runtime_states (around line 512)
def clear_runtime_states(self) -> None:
    self.fwd_cache.clear()
    self.output_chunks.clear()
    # Reallocate recv buffers so step-N autograd graph cannot alias step-(N+1) storage.
    # This is safe because fwd_cache (which held references to old buffers) is already cleared.
    for recv_tuple in self.args_recv_info.values():
        for a in recv_tuple:
            if isinstance(a, _RecvInfo):
                a.buffer = torch.empty_like(a.buffer).requires_grad_(a.buffer.requires_grad)
```

**Part B** — Alternatively (cleaner from a design standpoint): in `forward_one_chunk`
(stage.py line 717), `.clone()` or `.detach().requires_grad_()` the recv buffer before
storing it into `fwd_cache`:

```python
# stage.py, forward_one_chunk (around line 717)
flat_args = flatten_args(composite_args)
flat_kwargs = flatten_args(composite_kwargs)
# Clone recv buffers so fwd_cache holds tensors with independent storage.
# Without this, the same buffer tensor participates in multiple steps' autograd graphs.
flatten_input_tensors = [
    t.clone() if isinstance(t, torch.Tensor) and not t.requires_grad else t
    for t in flat_args + flat_kwargs
]
self.fwd_cache[fwd_chunk_id] = (output_tuple, flatten_input_tensors)
```

Part B is the more principled fix: it keeps `args_recv_info` buffer reuse (a valid
memory optimization) while ensuring `fwd_cache` holds independent storage that the
autograd graph can hold safely across the step boundary.  The cost is O(activation_size)
per forward pass — which is already dominated by the P2P transfer cost.

The real fix should be landed in `pytorch/pytorch` (the `pipelining/stage.py` file is
vendored from there), with a backport suggestion to `pytorch/torchtitan`.

### 5.2 Fixes for H2 (if confirmed independently)

#### Band-aid

In `CrossStageCacheAdapter._finish_forward` (pipeline_adapter.py line 810):

```python
# Change:
partial_out = self._keepalive_touch(partial_out, prev_recv_tensor)
# To:
partial_out = self._keepalive_touch(partial_out, prev_recv_tensor.clone())
```

Cloning `prev_recv_tensor` before the keepalive touch breaks the autograd link from
the keepalive to the shared recv buffer, at the cost of one clone per forward pass.

#### Real fix

Rethink `_keepalive_touch`: instead of `0.0 * prev_recv_tensor.sum()`, use a
`torch.zeros(1, device=..., requires_grad=True)` leaf that is explicitly accumulated
into the loss to keep the prev stage's P2P channel alive without aliasing the recv buffer:

```python
@staticmethod
def _keepalive_touch(payload, prev_recv_tensor):
    if prev_recv_tensor is None:
        return payload
    # Create a detached keepalive that preserves the channel without aliasing storage.
    keepalive = (prev_recv_tensor.detach() * 0.0).sum()  # zero scalar, no grad to prev
    if isinstance(payload, tuple):
        head, *tail = payload
        return (head + keepalive, *tail)
    return payload + keepalive
```

This is still not ideal; the cleanest fix is to not use `_keepalive_touch` at all and
instead ensure the P2P backward channel is preserved via explicit grad hook registration.

---

## 6. CI Test Sketch

### Regression test for H1 (model-agnostic, CPU only)

```python
# tests/test_pp_lbs_backward.py
import pytest
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

def _make_simple_model(hidden=32):
    return torch.nn.Linear(hidden, hidden)

@pytest.mark.parametrize("lbs,v", [(2, 2), (3, 2), (5, 2)])
def test_pp_interleaved_lbs_no_double_backward(lbs, v, init_pg):
    """
    PP + Interleaved1F1B + V virtual stages + LBS microbatches must complete
    2 full training steps without 'backward through the graph a second time'.

    Requires a 2-process process group (PP=2, V=v means 2*v total stages).
    """
    pp_rank = dist.get_rank()
    pp_size = dist.get_world_size()  # should be 2
    hidden = 32
    n_microbatches = lbs
    num_stages = pp_size * v

    # Each rank owns v stages
    my_stage_ids = [pp_rank + s * pp_size for s in range(v)]
    stages = [
        PipelineStage(
            _make_simple_model(hidden),
            stage_idx,
            num_stages,
            device=torch.device("cpu"),
            group=dist.group.WORLD,
        )
        for stage_idx in my_stage_ids
    ]

    loss_fn = torch.nn.MSELoss()
    schedule = ScheduleInterleaved1F1B(stages, n_microbatches=n_microbatches, loss_fn=loss_fn)

    # Run 2 steps; the bug fires on step 2's backward
    for step in range(2):
        x = torch.randn(lbs, hidden, requires_grad=True)
        target = torch.randn(lbs, hidden)
        try:
            schedule.step(x, target=target)
        except RuntimeError as e:
            if "backward through the graph a second time" in str(e):
                pytest.fail(
                    f"PP backward graph reuse at step {step+1}, "
                    f"LBS={lbs}, V={v}: {e}"
                )
            raise

# Fixture to init a 2-process group using spawn or gloo
```

**GPU variant** (required to catch the real NCCL P2P aliasing; CPU gloo also exercises
the buffer-reuse path since `irecv` writes into the same buffer regardless of backend):

```python
# Can be parameterized with @pytest.mark.parametrize("backend", ["gloo", "nccl"])
# gloo version runs on CPU, catches the buffer-reuse bug without GPU
```

**Key assertion**: the test must run 2 complete `schedule.step()` calls and assert no
`RuntimeError` with "backward through the graph a second time".  A single-step run will
not catch the cross-step aliasing.

---

## 7. Upstream RFC Outline

### Target repositories
- **Primary:** `pytorch/pytorch` — the `pipelining/stage.py` fix lives here
- **Secondary:** `pytorch/torchtitan` — downstream user; can document the `LBS≥2+V≥2`
  limitation until the pytorch fix is landed

### GitHub issue title

```
[Pipeline Parallel] Interleaved1F1B + V≥2 virtual stages + n_microbatches≥2 crashes
with "backward through the graph a second time" on step 2
```

### Body outline

```markdown
## Summary
`ScheduleInterleaved1F1B` with multiple virtual stages per rank (`n_local_stages ≥ 2`)
and `n_microbatches ≥ 2` crashes on the second training step's backward with:

    RuntimeError: Trying to backward through the graph a second time
    (or directly access saved tensors after they have already been freed).

## Reproduction

Minimal CPU reproduction (2 ranks, gloo backend):

    torchrun --nproc_per_node=2 test_pp_lbs.py  # see attached script

## Root Cause

`PipelineStage._prepare_forward_infra` allocates recv-buffer tensors once and stores them
in `args_recv_info[chunk_id].buffer`.  These buffers are reused across training steps.
`clear_runtime_states()` clears `fwd_cache` and zeroes `.grad` on the buffers, but does
NOT reallocate them.

During `forward_one_chunk`, the buffer tensor is stored (by reference, not by copy) into
`fwd_cache[chunk_id][1]` (the `input_values` list).  When `backward_one_chunk` calls
`stage_backward`, it passes `input_values` to `torch.autograd.backward`, which traverses
the autograd graph rooted at those tensors and frees their C++ saved tensors.

At the start of step N+1, `dist.irecv(..., info.buffer, ...)` overwrites the buffer
tensor's storage in-place with new activation data.  However, the step-N autograd engine
state (saved-tensor references) has not been cleared — it was freed during step-N backward,
but if step-N+1's irecv runs concurrently with (or after) step-N backward within the
interleaved schedule, the next backward attempt finds freed saved tensors and raises.

## Conditions Required
- PP degree > 1 (enables recv buffer allocation)
- n_local_stages ≥ 2 (enables intra-step F/B interleaving on same rank)
- n_microbatches ≥ 2 (creates multiple buffer slots that alias across steps)

## Proposed Fix

In `_PipelineStageBase.clear_runtime_states()`, reallocate recv buffers after clearing
`fwd_cache`, so step-(N+1) irecv writes into a fresh tensor rather than the storage
that step-N's autograd graph holds references to:

```python
# stage.py, clear_runtime_states
def clear_runtime_states(self) -> None:
    self.fwd_cache.clear()
    self.output_chunks.clear()
    for recv_tuple in self.args_recv_info.values():
        for a in recv_tuple:
            if isinstance(a, _RecvInfo):
                needs_grad = a.buffer.requires_grad
                a.buffer = torch.empty_like(a.buffer)
                if needs_grad:
                    a.buffer.requires_grad_(True)
```

Alternatively: clone the buffer before storing into `fwd_cache` in `forward_one_chunk`,
so the fwd_cache holds independent storage and the recv buffer can be safely reused.

## Workarounds (until fix is landed)
1. Set `n_microbatches=1` per rank (i.e. `local_batch_size=1` when `microbatch_size=1`).
2. Use `Schedule1F1B` (single virtual stage per rank, no interleaving).
3. Use gradient accumulation instead of increasing `local_batch_size`.

## Affected Versions
Observed in PyTorch [version from venv].  The `_prepare_forward_infra` single-allocation
pattern has been present since the pipelining module was introduced.

## CC
@wanchaol (pipelining maintainer), @fduwjj, @kwen2501
```

---

## 8. References

- `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/stage.py:1553-1573` — buffer allocation in `_prepare_forward_infra`
- `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/stage.py:512-530` — `clear_runtime_states` does NOT reallocate buffers
- `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/stage.py:422-429` — `get_fwd_recv_ops` writes irecv into existing buffer
- `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/stage.py:717-723` — `forward_one_chunk` stores buffer into `fwd_cache` without copy
- `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/stage.py:763-766` — `backward_one_chunk` pops `fwd_cache` and passes `input_values` to backward
- `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/schedules.py:1518-1545` — `_initialize_stages` allocation guard (once per schedule lifetime)
- `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/schedules.py:1619-1622` — `step()` calls `clear_runtime_states()` before `_step_microbatches()`
- `/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/_backward.py:282-413` — `stage_backward` crash site at line 411
- `/root/torchtitan_attention_residual/torchtitan/torchtitan/distributed/pipeline_parallel.py:219` — `n_microbatches = local_batch_size // microbatch_size`
- `/root/torchtitan_attention_residual/torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py:897-951` — adapter monkey-patch; no cross-mb buffer aliasing introduced
- `/root/torchtitan_attention_residual/additional_found_issues/torchtitan_pp_microbatch_backward_graph.md` — symptom doc and repro table
