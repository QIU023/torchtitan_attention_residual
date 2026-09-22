# PR title: [DO NOT review, stack on PR 4312] [Kimi K3] Pipeline parallelism: PP ranks park saved activations on a peer through the Mooncake Transfer Engine

Draft stacked on PR 4312. Review branch `pp_balance_review1` = `76f7d90df`, two commits on 4312's round-3 head `78be13c96`, which is five typed commits on upstream main `7349a2282`. The fork branch `k3_pp_balance` was force-pushed to this head on 2026-09-22; its previous head `080f44208` sat on the old 4312 head `de6f29514`.

Two correctness defects from the 2026-09-22 review of the draft branches are fixed, and both are covered by a CPU test:

- **The pool is partitioned, one span per source rank.** Every source used to build its allocator over the whole pool while the base came from the single destination entry, so two sources both started at offset zero, wrote the same destination bytes and fetched each other's: silently wrong gradients on any configuration with more than one source. The 09-16 smoke ran one source, so the path never executed.
- **A span is freed on the first read, and a second read raises.** The free happened inside `unpack`, which autograd does not guarantee to call once, and the free list had no double-free guard, so a schedule that consumes a saved tensor twice corrupted it. The allocator now tracks live offsets; `fetch` refuses a handle whose span is not live.

The abstraction findings are fixed too. The hooks install through the stage class the branch already owns: `AttnResPipelineStage.set_forward_context` runs the stage's forward inside a context the caller supplies, replacing the instance assignment over `forward_one_chunk`, and the stages hold the engine, so the side object on core's schedule and its pyrefly suppression are gone. `_stages_of`, which re-derived what the caller already held and returned an empty list when it could not, is gone; the caller passes its stages. The engine advertises the host it is reachable on and a free port instead of `127.0.0.1:{17000 + rank}`, which no peer on another node could reach and which two jobs on one node would collide on. The knobs are a `pp_balance` record on `KimiK3Model.Config`, because after #4810 the model owns its pipelining and there is no `pipelining_fn` to wrap, which also answers the finding that nothing in the branch constructed them. The environment switches and the `atexit` print are gone, `round_up` comes from `torchtitan/tools/utils.py`, and a `stats()` accessor plus a warn-once when the pool runs out replace the print.

Diff: 5 files, 484 added lines, 345 of them production non-blank, comment plus docstring 45 of those, 13.0 percent, from 17.9. CPU on this head: 56 passed (`pytest tests/unit_tests/cpu -k "kimi_k3 or pipeline"`), pyrefly clean on `torchtitan/models/kimi_k3`.

GPU (2026-09-22, 2 x RTX 5060 Ti over TCP, no HCA on this box, torch 2.15.0.dev20260906+cu130, mooncake-transfer-engine 0.3.13, seed 42, five steps, one warm compile cache): pp2 with four micro-batches, rank 0 parking on rank 1, reads `7.97110 / 7.28619 / 5.54323 / 5.32404 / 5.12824` on steps 1-5, identical to the same cell with the knobs unset on every step. Rank 0 parked 340 tensors, 420 MiB, and fetched all 340 back; 40 stayed local as not contiguous or not on the device, 460 were below the 1 MiB floor, and the 2048 MiB pool refused nothing. The RDMA path has still not run, for want of a box with an HCA. The hundred-step table on H100 comes before the draft leaves DO NOT review.

--- PASTE BEGIN ---

## Summary

Under an interleaved 1F1B schedule the resident activation load is uneven across pipeline ranks. With `pp_balance` set on the model config, the ranks it names run their stages' forwards under `saved_tensors_hooks` that copy what autograd saves into a pool on another rank and read it back in backward.

- `pp_balance.py`: the engine, a first-fit pool allocator, the hooks, and the knob record.
- `AttnResPipelineStage.set_forward_context`: the seam the hooks install through, so nothing assigns over a stage's method.
- A CPU test of the allocator: spans that do not overlap, alignment, coalescing on an out-of-order free, a full pool that refuses rather than overlapping, two sources that never share an offset, and a second free of one span that raises.

## Design

The copies are exact, so the loss is unchanged; what moves is where a saved tensor lives between the forward that produced it and the backward that reads it. A tensor that is not contiguous stays local, because rebuilding a non-contiguous save with a normalized layout keeps the values and changes the strides, and backward kernels reduce by layout.

Every source owns a span of the one pool: a per-rank allocator over the whole pool would hand two sources the same offset. A span is freed on the first read, and a second read of the same handle raises rather than corrupting the free list, so a schedule that consumes a saved tensor more than once is refused instead of trained wrongly.

Transfers go through the Mooncake Transfer Engine, which picks RDMA where an HCA exists and TCP where one does not; the pool and the staging buffer follow the same rule, device memory for RDMA and pinned host memory for TCP. `mooncake-transfer-engine` is an optional dependency with fla's standing: a box without it fails at install time with the package named, and the engine is not touched unless the knobs name a source rank.

## Test plan

- `pytest tests/unit_tests/cpu -k "kimi_k3 or pipeline" -q` (56 passed)
- pp2 with one source rank parking on the other, same seed and batch on one warm compile cache, against the same cell with the knobs unset: the loss is identical on every step, and the engine's counters show every parked tensor fetched back. The hundred-step table on H100, and the first RDMA run, go here.

--- PASTE END ---

Record, not for pasting: the 09-16 smoke on this stack is in the header above. The 09-05 measurement on `pp_balance_review1` `3bb3fc6cf` (previous adapter, 33-layer flavor, 5060 Ti, TCP): pp2 x vp4 bitwise off and on over 10 steps (`12.41967` / `7.47862` / `3.42131`), 1,440 tensors and 1,760 MiB parked and fetched back, 11,840 tensors below the 1 MiB floor kept local, rank 0's peak unchanged at that scale (about 176 MiB a step against 14 GiB). The keep-local mode that isolated the transfer from the allocator effects was a local diagnostic and is not in this branch.
