# Phase 3 adapter — session 2 log (2026-04-20 01:00 → 08:10)

Picks up from `handoff_status_20260420.md` (adapter plumbing OK, ran as
pure passthrough, delta not yet wired). This log covers the journey of
wiring the delta forward + a numerically-correct backward on 8× RTX
5090 PCIe.

## Outcome

**Current state**: delta forward + PP-autograd backward + a
`retain_graph=True` hack is working for the 150M smoke on 8 GPUs.
1000-step loss curve matches naive PP within bf16 tolerance (naive
step 1000 loss = 6.339, adapter = 6.346; Δ = 0.007). Forward bytes on
the wire match the static layout table. No deadlock.

Memory cost of the hack: +5 GiB peak on rank 7 (11.9 vs 6.9 GiB for
naive). Fits on 32 GiB cards for 150M; **will not scale** to 1.5B at
moderate M or to real 48B training. Proper fix is scoped below.

## Fork / workspace state

- Fork: `QIU023/torchtitan@attention_residual_dev`, HEAD `89868bd` +
  this retain_graph commit landing on top.
- Workspace: `QIU023/torchtitan_attention_residual@main`, HEAD
  `e866101` + a bump commit for the new fork HEAD.
- Tests: 36 / 36 green after the cleanup pass (forward numerics canary
  and backward param-grad equivalence both retained).

## Journey — what we tried, what failed, what finally worked

### 1. Static delta layout (Blocker-2, morning session)
Built `BlockLayoutTables` in `layout.py` that precomputes from
`(P, V, num_blocks, n_layers, layers_per_block)`:
- `commits_at(S)` — which block idx stage S commits (or none).
- `rank_cache_at_entry(R, v)` — what blocks rank R has cached when
  its v-th virtual stage is about to run.
- `delta_to_send(S)` — subset of (accumulated − receiver rank's
  cache) that gets stacked and sent forward.
- `consumer_stages_of(S)` — later stages that will cache a block
  committed at S (used for future grad accounting; now only telemetry).

Validated per-hop shapes match the 8-GPU runtime (per-rank dbg logs
confirmed stage 1 emits [1, B, T, D], stage 9 emits [3, B, T, D],
stage 8 emits [4, B, T, D], etc. — exactly what `delta_to_send`
predicts).

### 2. Custom grad P2P inside autograd.Function.backward — FAILED
Wrote `_SendBlockGradsBack` and `_RecvBlockGradsFromConsumers` to ship
per-block grads straight from autograd backward. NCCL timed out after
5 minutes: the autograd engine ran backward depth-first on a single
thread, and our `dist.isend(...wait())` blocked the engine while other
ranks' engines hadn't even reached their matching Function yet.
Interleaved1F1B's own `SEND_B/RECV_B` ops raced with ours on the same
process group. Deadlock.

### 3. Move NCCL out of autograd into post-`backward_one_chunk` sync hook — FAILED
Added `_flush_grad_sendback(mb)` invoked from `patched_bwd`'s finally
block. Still deadlocked: under Interleaved1F1B rank 0 and rank 7 reach
mb=0's backward at very different wall-clock moments, so rank 0's
flush posted ops with no matching peer ops; the schedule's next
`SEND_B` entangled with the pending P2P; NCCL watchdog fired a
collective timeout on rank 5.

### 4. Step-end flush — REJECTED before running
Idea: accumulate grads across all M mbs, do ONE synchronized
`batch_isend_irecv` after `pp_schedule.step()` returns. Fixes deadlock
(step boundary is torch-synchronized). Rejected for memory reasons:
per-mb cache + per-mb retained autograd graph × M in-flight mbs
blows up to TBs at real pre-training scale (M=32-128 for 48B).
Acceptable for a smoke but not the endgame.

### 5. Pure-autograd backward via PP SEND_B — PARTIALLY WORKS
Realized the Kimi §4.1 phrasing "backward is symmetric — send what
goes to the next stage directly" meant: **use PP's existing SEND_B**.
Cached blocks that came from a `recv_delta_tensor` already have a
live autograd link to that tensor; autograd routes their grad back
through PP SEND_B automatically. Deleted all custom autograd.Function
grad machinery; `pipeline_adapter.py` shrank from 1320 to 784 lines.

But double-backward error on own-rank cached commits: rank 0 caches
b0 from its v=0 commit, rank 0's v=1 stage traverses back into stage
0's forward graph during stage 8's backward, frees it. Then stage 0's
own backward (from PP SEND_B) tries to traverse the same graph again
and dies.

### 6. retain_graph=True on every stage's backward — WORKS (smoke-scale)
Added a narrow `torch.autograd.backward` monkey-patch inside
`patched_bwd` that forces `retain_graph=True` only during the delta
mode's primary backward. Same graph survives the second traversal.
Memory: +5 GiB on rank 7 for 150M. Numerics correct. **This is the
current committed state**.

## What needs fixing before scale-up — the _Local*_ plan

Replace the `retain_graph=True` hack with two thin local-only
`torch.autograd.Function` classes:

1. `_LocalCacheAugment(block, key)` — applied at producer emission
   when stage R commits block b that will be cached on rank R for
   later virtual stages.
   - `forward`: identity passthrough; record `key = (mb, stage, idx)`.
   - `backward(grad)`: read any captured grad from shared slot
     `rank_cache._captured_grads[key]`, return `grad + captured` as
     the input gradient. Stage R's wrapped params receive the
     summed contribution in their one normal backward pass.

2. `_LocalCacheCapture(block, key)` — applied at consumer read time
   when a later virtual stage on the same rank reads a cached
   own-rank commit.
   - `forward`: identity passthrough.
   - `backward(grad)`: write `grad` into
     `rank_cache._captured_grads[key]` (accumulating for V>2) and
     return `None` → autograd STOPS; stage R's forward graph is
     never traversed by the consumer's backward.

Recv-originated cached blocks (those that came in via PP
`recv_delta_tensor` at an earlier virtual stage on this rank) do NOT
get wrapped — their grad still flows back through PP SEND_B
naturally.

Why this avoids every past trap:

| Past trap | Why it doesn't repeat |
|---|---|
| Custom NCCL in autograd | Zero NCCL. Two Functions only touch a Python dict. |
| Step-end memory blowup | Only one `B×T×D` tensor per captured slot; graph lives single-backward window so peak memory ≈ naive PP. |
| retain_graph blowup | Each stage's graph traversed exactly once; `retain_graph=False`. |
| Double-backward | Capture returns None → producer's graph never traversed by consumer path. |

The pattern is standard PyTorch (same shape as `torch.utils.checkpoint`'s
non-reentrant machinery and FSDP's `register_post_accumulate_grad_hook`).

**Gate before integrating**: write 3-4 CPU-only tests confirming
- Augment.backward correctly sums incoming + captured grads,
- Capture.backward returns None and stops downstream propagation,
- multi-consumer slot accumulation works for V>2,
- producer's `param.grad` matches naive autograd after the full dance.

Only once those pass, swap the retain_graph hack for the Functions.

## Files touched this session (fork)

- `torchtitan/experiments/attn_res/pipeline_adapter.py` — delta
  forward, schedule guard, shape-inference delta reshape, cleanup,
  retain_graph hack.
- `torchtitan/experiments/attn_res/layout.py` — new; BlockLayoutTables.
- `torchtitan/experiments/attn_res/tests/test_pipeline_adapter.py` —
  layout coverage, forward-delta numerics, backward-param-grad
  equivalence; removed dead tests from rejected approaches.

## Files touched (workspace)

- Submodule pointer bumps.
- This file.

## Launch recipe for next session

Preconditions: tokenizer under
`torchtitan/assets/hf/Llama-3.1-8B/`, C4 shards prefetched under
`$HF_HOME`, env `source /venv/main/bin/activate`.

```bash
# adapter run (current retain_graph-based)
rm -rf phase3/runs/pp8_adapter
bash phase3/launch_8gpu_adapter.sh
# compare to naive
python phase3/compare_pp_vs_single.py \
    --single phase2/runs/attn_res/tb \
    --pp    phase3/runs/pp8_naive/tb \
    --pp_cached phase3/runs/pp8_adapter/tb
```

## Residual risks the retain_graph hack carries

1. `torch.autograd.backward` monkey-patch is process-global for the
   duration of one `backward_one_chunk` call. Not thread-safe if
   anyone else is calling `torch.autograd.backward` concurrently (no
   one is in this codebase, but worth knowing).
2. Memory doesn't scale — see above.
3. Every stage's `retain_graph=True` means every stage's graph lives
   until its rank cache evicts. Under V>2 that's more memory still.

## Next session todo

- Implement `_LocalCacheAugment` + `_LocalCacheCapture` per above.
- CPU tests first.
- Swap the retain_graph hack, rerun 8-GPU 1000-step, confirm memory
  drops back to naive-PP baseline and loss still matches naive.
- After that, scale-up config (1.5B or 2B) on 8-GPU to validate the
  "cross-stage caching under PCIe" headline benchmark.
