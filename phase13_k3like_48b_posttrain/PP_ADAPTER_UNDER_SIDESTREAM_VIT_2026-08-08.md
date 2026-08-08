# The PP adapter under a side-stream ViT: what its safety rests on

Full re-read of `pipeline_adapter.py` (1530 lines) against the decision to run the
vision encoder concurrently with the text pipeline on its own communicator.

**Conclusion first: the adapter needs no code change.** But its safety argument rests
on three properties, only one of which is the "PP owns all NCCL" line everyone
quotes, and one of the other two fails SILENTLY rather than loudly. That one is the
real constraint on the design.

---

## What the adapter actually does

Two bridges carry AttnRes block state, and neither is a collective:

1. **Cross-rank** -- the delta of newly committed blocks rides inside the stage's
   own output tuple, so PP's `SEND`/`RECV` transports it and PP's `SEND_B` returns
   the gradient. The adapter adds no transport of its own.
2. **Same-rank across virtual stages** -- a plain Python dict (`RankLocalCache`) plus
   autograd plumbing: a tensor grad hook on the producer's block
   (`_install_augment_hook`), a **detached** copy in the cache, and
   `_LocalCacheCapture` on the consumer side whose backward deposits the gradient in
   a slot and stops. The detach is load-bearing: a hook plus
   `Function.backward -> None` was tried and did NOT survive real PP + FSDP + AC
   rerun.

Verified by reading: **`grep` for `cuda.stream|current_stream|record_stream|Event(`
in the adapter returns nothing.** It has no stream discipline at all, because it has
never needed any -- everything it does happens on whatever stream PP is already using.

---

## Property 1: "PP owns all NCCL" -- what it buys, and what it does not

The claim in the module docstring is:

> Both bridge mechanisms are pure local Python + a dict -- zero NCCL, zero cross-rank
> state. The grad walks backwards hop-by-hop along the same PP stage chain that
> forward uses (PP owns all NCCL, so no deadlock risk)

What it buys: NCCL requires that operations on a given communicator be **issued in
the same order on every participating rank**. If the only issuer is the schedule, that
order is consistent by construction, and no reasoning about interleaving is needed.

**A separate process group for the vision exchange preserves exactly this.** A
distinct PG is a distinct communicator, so its operations are ordered only among
themselves; PP's ordering argument is untouched. The invariant weakens from "PP owns
all NCCL" to "PP owns all NCCL **on the PP communicator**", which is what the argument
actually needed.

What it does NOT buy, and this is the part to design against: two communicators can
still deadlock on a **cyclic wait**. If rank A blocks on the vision PG while rank B
blocks on the PP PG and each is waiting for the other to move, no single
communicator's ordering was violated and the job still hangs until the watchdog.

The avoidance rule is already in this codebase, learned the hard way in the
multimodal wrapper: **gate a collective on the MESH, never on the data.**
`_exchange_sentinel_counts` is gated on `cp_world_size() > 1` rather than on whether a
rank has images, and `_keep_tower_alive` exists precisely because "a rank that skips
the tower does not issue the all-gather and its peers wait until the NCCL watchdog
fires". The vision exchange must obey the same rule: **every rank in the vision PG
must reach every vision collective, in the same order, regardless of what PP is doing
or what the batch contains.** In practice that means the exchange schedule is a
function of the micro-batch count and the mesh, decided before the step, not of
per-rank progress.

---

## Property 2: the mb index is THREAD-LOCAL, and violating it fails silently

This is the constraint the re-read was worth doing for.

The adapter keys its per-microbatch cache by the schedule-owned chunk id, stashed by
monkey-patching `forward_one_chunk` / `backward_one_chunk`:

```python
_mb_state = threading.local()          # module level
def _set_mb_index(adapter_key, mb_index): ...   # writes _mb_state.indices
```

and the docstring's justification is "forward and backward run synchronously on the
same thread, so autograd hooks that fire during backward can read the mb index".

Now the failure mode. `CrossStageCacheAdapter.forward` begins:

```python
if _current_mb_index(self._adapter_key()) is None:
    return self._forward_shape_inference(*args, **kwargs)
```

**A missing mb index is interpreted as "this call is PP's shape inference."** It is not
an error. So if any part of the model were driven from a different Python thread --
where the thread-local is simply absent -- the adapter would take the shape-inference
path: a differently shaped output, no cache append, no exception. A silent wrong
answer.

Consequence for the design, and it is not negotiable:

> **The vision encoder must run on the same Python thread as the schedule, using a
> separate CUDA stream for overlap. Not a worker thread.**

CUDA streams give the concurrency; threads are what break the keying. A background
thread would appear to work -- shapes flow, nothing raises -- and be wrong.

(`_current_mb` itself asserts non-None, so a path that reached it from another thread
would raise. But `forward` checks first and diverts, so the assert is unreachable for
exactly the case that matters.)

---

## Property 3: no stream discipline exists yet, so all of it is new

The adapter never touches streams. Under a side-stream ViT, three things become the
new design's responsibility, none of them the adapter's:

* **Cross-stream tensor lifetime.** A tensor produced on the vision stream and read on
  the default stream needs `record_stream` or an event, or the caching allocator can
  hand its memory to another allocation while the consumer still reads it. This is a
  correctness hazard, not a performance one.
* **FSDP2's own collectives for the tower.** FSDP issues the tower's all-gather from
  its pre-forward hook, on whatever stream is current. Running the tower on a side
  stream puts those collectives there too, and FSDP's internal streams and events
  assume their own ordering. This needs checking against FSDP rather than assumed --
  it is the interaction most likely to produce something that works at pp2 and hangs
  at pp8.
* **The AttnRes delta is unaffected**, because it rides PP's output tuple on PP's
  stream. Its shapes and content do not change.

---

## What does NOT change

* The adapter's code. Both bridges are rank-local or ride PP's transport; a vision
  communicator is orthogonal to both.
* The delta's content and shape, so PP's shape inference is unaffected.
* "each stage's forward graph is traversed exactly once per mb", so the peak-memory
  equality with naive PP is preserved.
* The detach-based double-backward fix, which is about autograd structure and knows
  nothing about streams.

---

## Verification plan, as requested

**Configuration: pp8 x vp4, multimodal.** That is the pressure test the adapter was
validated on, and it is where a stream or ordering mistake shows up -- the pp2 case
has too few stages to expose an ordering cycle.

**Toggle: one parameter, on/off, same binary.** DEP enabled versus disabled in
otherwise identical runs. `KIMI_VIT_DEP` already works this way and defaults off.

**Measurement: total step latency, A/B on the same box.** Worth being precise about
why this is legitimate here when I argued earlier against timing. The earlier argument
was against extrapolating an *absolute* overlap fraction from PCIe hardware to the
report's fabric -- that does not extrapolate. A *relative* A/B of two arms on the
same box measures whether this implementation hides work on this hardware, which is a
different and answerable question. Both statements have to travel together:

* "DEP reduced step latency by X% at pp8xvp4 on 8x5060Ti/PCIe" -- reportable.
* "DEP hides most of the ViT computation" -- not reportable from that number, because
  the vision share of a step here is not the vision share at 2.8T. Measured from the
  configs: one ViT forward is 25.2 text layers on the debug flavor and 0.057 of one
  layer on `2p8t_vl`, ~440x apart.

**Correctness gate first, before any latency number is quoted.** The A/B must be
numerically exact from a shared seed checkpoint, the way the stage split already is:
step-1 loss identical, grad_norm within the bf16 floor. A latency win on a run whose
numerics moved is not a win.

**And the negative control that the structural analysis makes necessary:** with DEP
off, the same run must NOT show the improvement. That sounds trivial, but the current
DEP already fails a related control -- bubble count is identical with and without it
-- so "the toggle changes something measurable" is exactly the claim in question.
