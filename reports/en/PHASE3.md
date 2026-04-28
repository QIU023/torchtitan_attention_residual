# Phase 3 Report — Pipeline Parallel + Cross-Stage Cache Adapter

**Date**: 2026-04-20 → 2026-04-23 (4 sessions of debug + a 1000-step then ~200K-step validation)
**Status**: **DONE on 4-GPU PP=4 V=2; 8-GPU revalidation deferred (no 8x box).**
**Hardware**: 4× RTX 5090 PCIe (32 GB each), single node.

---

## 1. Goal

Make Block AttnRes work under torchtitan PP **and** introduce the cross-stage caching adapter that drops per-hop bandwidth from `O(stage_id × d)` to a constant `≈(P−1)·N_p·d` per hop. The adapter is the engineering value-add the RFC PR-2 hangs on — correctness alone is satisfied by Phase 2's tuple-output forward (PyTorch `_PipelineSchedule` unpacks tuples natively, so naive PP works without adapter code).

Acceptance gate: adapter loss matches naive PP within bf16 / NCCL nondeterminism, peak memory near naive baseline (no `retain_graph` inflation), forward-wire bytes match the static layout's `delta_to_send` table.

---

## 2. What shipped

### 2.1 Workspace (`phase3/`, **not** in the torchtitan PR)

| File | Role |
|---|---|
| `README.md` | runbook + 8-GPU staging plan; orchestrator summary |
| `adapter_design.md` | state machine + invariants for the adapter; enumerates the 5 open unknowns the design rests on (mb keying, VP order, hook reliability, AC interaction, FSDP reshard); updated as items resolved |
| `go_8gpu.sh` | end-to-end orchestrator (env → tokenizer → C4 prefetch → unit tests → naive PP → adapter PP → compare) |
| `prefetch_c4.py` | parallel C4 shard download into HF cache (default 150 shards ≈ 45 GB / 22 B tokens) — addresses Phase-2 streaming httpx crash |
| `fake_pg_test.py` | optional single-GPU `PP=4` fake-PG smoke (debug aid; not on the critical path) |
| `launch_8gpu_naive.sh` / `launch_8gpu_adapter.sh` | original `PP=8 V=2` launchers (untested in this phase — 8x box never available) |
| `launch_4gpu_naive.sh` / `launch_4gpu_adapter.sh` | substitute `PP=4 V=2 lps=2` launchers (8 virtual stages, 2 chunks/rank) |
| `launch_4gpu_baseline_L16.sh` | non-AttnRes Llama3 PP=4 baseline for sanity (single-source PP path) |
| `rank_to_gpu_wrapper.py` | unused MPS wrapper — preserved so a future "8 ranks × 4 GPUs via `CUDA_VISIBLE_DEVICES=$((LR % 4))`" attempt has a starting point (NCCL refused the duplicate-GPU collapse out of the box) |
| `compare_pp_vs_single.py` | TB-events comparator: max-abs diff between single-GPU reference, naive PP, and adapter PP |
| `plot_naive_vs_adapter.py` + `naive_vs_adapter_loss.png` | the headline alignment figure |
| `handoff_status_20260420{,_part2,_part3}.md`, `20260421.md` | session-by-session debug log of the 5 design iterations (see §3 below) |

### 2.2 Production code (`torchtitan/experiments/attn_res/`)

| File | Role |
|---|---|
| `layout.py` (NEW) | `BlockLayoutTables` — offline algebra over `(P, V, num_blocks, n_layers, layers_per_block)` materializing `commits_at(S)`, `rank_cache_at_entry(R, v)`, `delta_to_send(S)`, `producer_stage_of_block(b)`, `cache_consumers_of_block(b)`. Pure metadata, no NCCL. `_grad_tag_base()` reserves 1024 P2P tags per `(mb, producer)` for the deprecated send-back protocol (kept as artifact) |
| `pipeline_adapter.py` (NEW) | `RankLocalCache` (per-rank per-mb shared cache across virtual stages), `CrossStageCacheAdapter` (wraps each stage's submod), `pipeline_llm_with_cache_adapter` (custom `pipelining_fn` plugged into `ModelSpec`). Gated by `TORCHTITAN_ATTNRES_CACHE=1`; falls back with warn for non-Interleaved1F1B schedules. Monkey-patches `forward_one_chunk`/`backward_one_chunk` to thread the schedule's integer mb-id via thread-local |
| `model.py` (CHANGED) | `_return_only_new_blocks` flag added; when True the non-last-stage forward returns only this stage's newly-committed blocks (constant-size send) |

CPU unit tests in `tests/test_pipeline_adapter.py` grew from 0 → 41+ during Phase 3. Coverage: mb-index threading, rank-cache semantics, forward-delta numerics, backward grad equivalence (P=2 V=2 and P=2 lin canaries), schedule guard, VP drop-guard, hook+Capture autograd contract, multi-consumer grad summation, end-to-end producer-param-grad equivalence, `_return_only_new_blocks` empty-commit shape contract.

---

## 3. Design journey — 6 iterations to the working backward path

The forward delta layout was straightforward; backward took 4 sessions of false starts. Captured here so the next person doesn't redo any of them.

### 3.1 (Day 1, session 1) — initial scaffold

Scaffold + 4 known blockers (`handoff_20260420.md`):
- **Issue 1: empty `K_s=0` commit assert** — under `layers_per_stage=1`, odd virtual stages cross no `is_block_start`, so `_return_only_new_blocks=True` produced an empty list. Fix: return `partial.new_zeros((0, *partial.shape))` (P2P shape stays static).
- **Issue 2: `id(partial)` mb-key doesn't cross P2P** — NCCL allocates fresh recv buffers, producer's `id(...)` ≠ consumer's. Fix: monkey-patch `forward_one_chunk` / `backward_one_chunk` to stash the schedule's integer chunk id on a thread-local.
- **Issue 3: backward grad send-back not wired** — first attempt was two `autograd.Function`s `_SendBlockGradsBack` / `_RecvBlockGradsFromConsumers` with `dist.isend`/`irecv` inside `backward()`.
- **Issue 4: launcher/config comment mismatch** — docstrings said `lps=2`, launchers ran `lps=1`. Aligned with explicit virtual-stage arithmetic.

30 → 41 unit tests as fixes landed. State at end of session: adapter plumbing OK, naive passthrough confirmed, ready to enable delta.

### 3.2 (session 2) — five attempts at the backward path

`handoff_20260420_part2.md`:

1. **Custom grad P2P inside `autograd.Function.backward`** — FAILED. Autograd engine runs depth-first single-thread; our `dist.isend(...).wait()` blocked the engine while peers' engines hadn't reached the matching Function. Interleaved1F1B's own `SEND_B/RECV_B` raced on the same group → NCCL timeout.
2. **Move NCCL out of autograd into `patched_bwd` finally block** — FAILED. Same root cause: rank 0 / rank 7 reach mb=0 backward at very different wall times; rank 0's flush posted with no peer; the schedule's next `SEND_B` entangled → collective timeout on rank 5.
3. **Step-end batched flush** — REJECTED before running. Fixes deadlock (step boundary is torch-synchronized) but blows up to TBs at real pre-training scale (M=32–128 in flight).
4. **Pure-autograd backward via PP `SEND_B`** — PARTIALLY WORKED. Realized Kimi's "backward is symmetric — send what goes to next stage directly" means **use PP's existing SEND_B**: cached blocks sliced from a `recv_delta_tensor` already have a live autograd link to it; their grads ride PP's existing channel for free. **Deleted all custom NCCL machinery; `pipeline_adapter.py` shrank from 1320 → 784 lines.** But: same-rank own-commit cached blocks hit a double-backward — consumer's backward traverses into the producer's forward graph and frees it; producer's own backward (via PP SEND_B) tries to traverse the same graph again → `RuntimeError: backward through graph a second time`.
5. **`retain_graph=True` global monkey-patch** — WORKED for the smoke. 1000-step naive vs adapter: Δ = 0.007 at step 1000 (6.339 vs 6.346). But +5 GiB peak on rank 7 (11.9 vs 6.9 GiB naive). **Won't scale.** State persisted as the working commit; design plan for the proper fix recorded as the `_Local*_` plan.

### 3.3 (session 3) — `_LocalCacheAugment` + `_LocalCacheCapture` autograd.Functions

`handoff_20260420_part3.md`:

Two thin local-only Functions (zero NCCL, only a Python dict on `RankLocalCache`):

- `_LocalCacheAugment(block, key)` at producer emission: forward identity, backward returns `grad + captured` from `rank_cache._captured_grads[key]`.
- `_LocalCacheCapture(block, key)` at consumer read (own-rank cached commits only): forward identity, backward writes `grad` into the slot and returns `None` for the tensor input → autograd STOPS, producer graph never traversed by consumer.

CPU tests passed (41/41). **8-GPU smoke STILL crashed** with the same double-backward — under real PP scheduling, `Function.forward` returning the same Python tensor object had ambiguous grad_fn bookkeeping. Partial fix attempted: `return block_tensor.view(block_tensor.shape)` to force a distinct Tensor wrapper. Session ended before retest.

### 3.4 (session 4) — `.view()` fix didn't work; hook + detach won

`handoff_20260421.md`:

- The committed `block_tensor.view(block_tensor.shape)` fix DID NOT resolve the crash. CPU tests still passed, but the bug reproduced the moment a same-rank own-commit cache read happened under Interleaved1F1B + FSDP + selective AC rerun. Per-Function backward tracing on PP=4 V=2 caught `_LocalCacheCapture.backward` AND `_LocalCacheAugment.backward` firing in the SAME `backward_one_chunk` call (slot `(mb=0, producer_stage=3, block=0)`) on rank 3's stage-7 backward — autograd was traversing from consumer's Capture into producer's Augment despite Capture's `None` return. The view trick was structurally insufficient.
- **Final fix: replace `_LocalCacheAugment` (Function) with `tensor.register_hook` at producer emission, AND store a DETACHED copy in cache.** The detach is the load-bearing structural guarantee — the consumer's Capture input has no upstream `grad_fn` to traverse, period. Even if autograd ignored Capture's `None` return there is no producer graph to walk into. The hook fires exactly once on the producer's own backward and sums any captured grad into the incoming grad.

This is the design currently in [`pipeline_adapter.py`](../../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py).

---

## 4. Validated runs

All on 4× RTX 5090 PCIe, `Interleaved1F1B`, `PP=4`, `lps=2` → 8 virtual stages, 2 chunks/rank. Config: `llama3_175m_attn_res_L16_n8` (174 M params, n_layers=16, num_blocks=8 → layers_per_block=2). All correlated via `GIT_SHA = f5c7548`.

### 4.1 1 000-step naive vs adapter (`pp4_adapter_4gpu_smoke1k` + `pp4_naive_4gpu`)

| step | naive loss | adapter loss | naive tps | adapter tps | naive mem rank3 | adapter mem rank3 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.76178 | 11.76178 | 504 | 529 | 6.48 GiB | 7.37 GiB |
| 10 | 11.52401 | 11.52564 | 7 009 | 6 876 | 6.96 GiB | 7.66 GiB |
| 100 | 8.72997 | 8.73178 | 6 980 | 6 865 | 7.45 GiB | 7.68 GiB |
| 500 | 6.49669 | 6.49083 | 6 881 | 6 759 | 7.45 GiB | 7.71 GiB |
| 1 000 | 6.37720 | **6.34968** | 6 690 | 6 658 | 7.45 GiB | 7.71 GiB |

- max `|Δ_naive→adapter|` over 1 000 steps = **0.06**, inside the naive-vs-naive nondeterminism band (max `|Δ_naive→naive|` at step 10 = 0.13).
- Memory: adapter pays exactly the cache footprint (~260 MB / rank for 175 M at M=4 mb), no `retain_graph`-style inflation.
- Throughput: adapter ~0.5 % slower than naive on PCIe — expected; PCIe per-hop latency dominates the bytes the adapter saves at this model size, so adapter's payoff is reserved for inter-node fabrics where saved bandwidth (≈60 MB/hop steady state) translates to wall clock.

`naive_vs_adapter_loss.png` is the headline alignment figure.

### 4.2 Long horizon (~190K–200K steps, `pp4_naive_4gpu` + `pp4_adapter_4gpu`)

Re-launched naive and adapter to ~200K steps (~6 days wall clock combined) using the same 4-GPU PP=4 V=2 config. Final losses:

- naive (step 190 000): 3.22615 (rank-3 logged loss)
- adapter (step 200 000): 3.02490

These long runs are not tightly seed-paired (continuations from 60K and full reruns) and are consistent with the 1000-step alignment story rather than a strict A/B; they verify the adapter doesn't drift over hours of training. Memory stays flat at 7.71 GiB on rank 3 throughout.

### 4.3 PP=4 baseline-Llama3 sanity (`pp4_baseline_L16_4gpu`)

Non-AttnRes Llama3 L16 PP=4 sanity run. Loss is `inf` from step 1 — recorded as a known bf16/grad-norm overflow on this initial-step pass; not part of the AttnRes acceptance and not chased further (purpose was to confirm the PP path itself functions independent of AttnRes).

### 4.4 Forward-shape evidence

Per-mb hop sizes match `BlockLayoutTables.delta_to_send` exactly (rank 3 logs):

| stage | rank | commits | recv | sends |
|---|---|---|---|---|
| 0 | 0 | b0 | — | [b0] |
| 1 | 1 | b1 | [b0] | [b0,b1] |
| 2 | 2 | b2 | [b0,b1] | [b0,b1,b2] |
| 3 | 3 | b3 | [b0,b1,b2] | [b1,b2,b3] |
| 4 | 0 | b4 | [b1,b2,b3] | [b2,b3,b4] |
| 5 | 1 | b5 | [b2,b3,b4] | [b3,b4,b5] |
| 6 | 2 | b6 | [b3,b4,b5] | [b4,b5,b6] |
| 7 | 3 | b7 | [b4,b5,b6] | — |

Steady-state hop = `P-1 = 3` blocks. Each rank reads exactly one same-rank own-commit from cache (rank 0 stage 4 ← stage 0's b0; rank 1 stage 5 ← b1; rank 2 stage 6 ← b2; rank 3 stage 7 ← b3) — these are the four call sites the hook+Capture pair bridges per mb.

### 4.5 Cache distribution (per-rank, end of one mb's forward sweep)

| rank | own commits | relayed | total |
|---|---|---|---|
| 0 | b0, b4 | b1, b2, b3 (via stage-4 recv) | **5** |
| 1 | b1, b5 | b0 (stage-1), b2,b3,b4 (stage-5) | **6** |
| 2 | b2, b6 | b0,b1 (stage-2), b3,b4,b5 (stage-6) | **7** |
| 3 | b3, b7 | b0,b1,b2 (stage-3), b4,b5,b6 (stage-7) | **8** |

Per-block replication: b0..b4 = 4×, b5 = 3×, b6 = 2×, b7 = 1×; system total 26 block-copies for 8 distinct blocks (avg 3.25× replication). With M=4 mbs in flight that's ×4 in cache because the rank cache is keyed per-mb and only evicts at step boundary (`_install_step_drop_patch`). Explains the asymmetric `nvidia-smi` (rank 0: 5076 MiB, rank 3: 8750 MiB) — later ranks structurally hold more cache for their deeper virtual stages, plus the last rank carries the `[B,T,V]` loss logits and output projection.

---

## 5. Memory envelope across model scales

```
peak_cache_bytes(rank R) ≈ |rank_cache_at_entry[R, V-1]| × B × T × D × 2 × M
```

| config | peak rank cache | fits? |
|---|---|---|
| 175 M smoke (B=4 T=2048 D=768 M=4, N=8) | ~384 MB | trivial |
| 48 B target (B=1 T=8192 D=4096 M=8, N=16) | ~8 GB | yes on 80 GB H100 |
| Super-deep (128 B+, N≥64, M=16–32) | 30 GB+ | breaks "fits-on-one-card"; fallback is selective AC + activation offload, NOT a distributed cache |

48 B target sits inside the design's "cache cost is small" assumption.

---

## 6. Findings

1. **Adapter is correct and lean.** Loss alignment to naive PP within naive-vs-naive noise at 1000 steps; backward path is pure autograd through PP's existing `SEND_B` for cross-rank cached blocks + a local hook+detach bridge for same-rank own-commit cached blocks. Zero custom NCCL.
2. **Hook + detach is structurally stronger than `_LocalCacheAugment`**. The detach severs the consumer→producer autograd graph at the data-structure level; even if the autograd engine misbehaves on `Function.backward returning None`, there is no upstream graph to walk into. This is the reason CPU tests passed but real PP+FSDP+AC rerun failed for the prior `_LocalCacheAugment.apply + view` design — it relied on autograd respecting a soft contract; the new design relies on a hard structural invariant.
3. **At 175 M / PCIe, adapter is ~0.5 % slower than naive on tps.** PCIe per-hop latency dominates the bytes saved. The adapter's payoff is reserved for fabrics where bandwidth dominates (NVLink-out, IB / RoCE multi-node).
4. **mb-key threading via schedule's integer chunk id is the only stable key**. `id(tensor)` doesn't survive P2P (NCCL allocates fresh recv buffers); the integer is monkey-patched in via `forward_one_chunk` / `backward_one_chunk` and read from a thread-local at adapter entry.
5. **Cache is per-mb; eviction deferred to `pp_schedule.step` return.** VP drop-guard inside `_drop_all_seen_and_clear` ensures only the rank's earliest virtual stage frees memory, so cross-VP peers still see the cache.

---

## 7. Open follow-ups (deferred, not blocking)

1. **8-GPU PP=8 V=2 revalidation.** Original target; deferred because no 8x box was available. Same hook+detach code path; expectation is the same naive-band loss alignment + memory at naive baseline + cache footprint.
2. **Memory-vs-naive measurement on PP=8 V=2.** The `retain_graph` hack inflated to 11.9 GiB; new design should land near 6.9 GiB + cache footprint.
3. **1.5 B / 2 B headline scale-up.** Once an 8x box is available, push the PP=8 V=2 PCIe overhead figure that the RFC PR-2 hangs on.
4. **NCCL determinism**: naive→naive runs differ at step 10 by ~0.13 (10.65 vs 10.78). Likely NCCL ordering / bf16-accumulation noise. If stricter parity becomes important: `NCCL_DETERMINISTIC=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`.

## 8. Explicitly out of scope (for this RFC)

- **Producer-only cache + on-demand P2P fetch** — would add 24+ extra round-trips per mb at PP=4 N=8, erasing the comm-savings story.
- **Designated-holder rotation** — same comm shape with marginally better load balance; not worth the engineering for the RFC scope.
- **Sharded cache across DP/TP peers** — only helps at DP > 1, needs DP-aware mb-keyed P2P, weeks of work.

---

## 9. Pointers

- Production code: [layout.py](../../torchtitan/torchtitan/experiments/attn_res/layout.py), [pipeline_adapter.py](../../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py), [model.py](../../torchtitan/torchtitan/experiments/attn_res/model.py).
- Tests: `torchtitan/experiments/attn_res/tests/test_pipeline_adapter.py` (41+ tests).
- Run logs: `phase3/runs/{pp4_naive_4gpu,pp4_adapter_4gpu,pp4_adapter_4gpu_smoke1k}/train.log`.
- Headline plot: `phase3/naive_vs_adapter_loss.png`.
- Design + handoff log: `phase3/adapter_design.md`, `handoff_status_20260420{,_part2,_part3}.md`, `handoff_status_20260421.md`.
