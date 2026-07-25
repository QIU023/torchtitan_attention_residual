# PR #18 — verl torchtitan engine: context-parallel path is broken for every model

**Target repo**: `volcengine/verl`
**Target file**: `verl/workers/engine/torchtitan/transformer_impl.py`
**Fork reference**: verl `kimi_k3_integration`, commit `60b185fe` (5 defects; **3 of the 5 are model-agnostic and are what this PR proposes**)
**Upstream audit (2026-07-25)**: `upstream/main` @ `983cb0f2` still has the unfixed `_get_data_parallel_mesh` (returns the `fsdp` mesh) and no `positions` bridging. **Not obsoleted.**
**Effort**: ~0.5 day (3 hunks + a dp1+cp2 SFT smoke).
**Risk**: low for the mesh and `positions` fixes (they only change behaviour when `cp > 1`, which currently cannot work at all). Medium for the logits gather — it is correct but memory-hungry at long context; see "Open design point".

---

## Suggested PR title

> [engine] fix: torchtitan engine context parallelism — sampler mesh includes cp, `positions` missing from the CP input prep, and seq-sharded logits reach a full-sequence loss

## Suggested PR body

### Summary

The torchtitan engine exposes `context_parallel_size`, but the CP path has
never worked end to end. Enabling `cp > 1` hits three model-agnostic defects:

**1. The dataloader mesh includes the CP axis** (`_get_data_parallel_mesh`).
torchtitan's `"fsdp"` mesh is `dp_shard x cp` — FSDP folds cp into its
gradient-reduction axis — so it is *not* the data-loading axis. All cp ranks of
one dp group must receive the **same** batch; the trainer seq-shards it across
cp afterwards. Using `"fsdp"` makes rank `cp_i` draw sample shard `i`:

- at `dp_shard=1, cp=2` the `DistributedSampler` raises (`rank 1` not in
  `[0, 0]`),
- with real dp it does not raise — it silently trains on wrong batches, and
  each cp rank sees a different sequence, so the seq-shard reassembly is
  meaningless.

Fix: use the `"batch"` mesh (`dp_replicate x dp_shard`), which is exactly the
sampler axis. Legacy fallbacks are kept but guarded to `cp == 1`, where
`fsdp == dp_shard` and they were correct.

**2. `prepare_context_parallel_input` needs `positions` in `extra_kwargs`.**
It seq-shards `positions` alongside inputs and labels, but this engine keeps
positions in `extra_inputs`, so the call raises `KeyError` — a latent bug that
only surfaces once anyone turns CP on. Fix: bridge `positions` across the call.

**3. Seq-sharded logits reach a full-sequence loss path.** verl's loss code
(nested / no-padding log-prob handling) assumes full sequences. With CP the
model returns logits for `T/cp` positions. Fix: keep labels full length and
all-gather the logits over the cp group after the model call, using
`torch.distributed.nn.functional.all_gather` so the backward is a
reduce-scatter (differentiable).

### Test plan

SFT on the torchtitan engine, `dp1 + cp2` vs single GPU, deterministic seed:
step-1 loss must match within the bf16 band and the trajectory must track. On
our fork (`kimi_linear_194m` fixture, 2x RTX 5060 Ti): 12.143 vs 12.149
single-GPU, full epoch, `dp2` baseline unaffected; and `dp2 x cp2` on 4 ranks:
12.133 -> 3.818.

### Open design point (reviewers should weigh in)

The logits gather materialises `[B, T, V]` on every cp rank, which defeats part
of the memory win CP exists for. The right long-term fix is a **sharded-loss**
variant: compute the loss per cp shard and reduce, never gathering the full
logits. That is a larger change to verl's loss contract, so this PR takes the
correct-but-costly route and leaves a `TODO`. Happy to split the gather behind
a flag if maintainers prefer.

---

## NOT proposed here (fork-local, needs an upstream design first)

`60b185fe` contains two more fixes that are gated on `torchtitan_name ==
"kimi_k3"` and should **not** be filed as-is — a model-name check does not
belong in the shared engine:

- **`context_parallel_load_balancer=None`**: models whose CP is implemented
  *inside the module* (Ulysses-style, as our kimi_k3 does) require contiguous
  rank-ordered sequence shards. torchtitan's default `headtail` balancer
  permutes the sequence before sharding, which for such a model means
  future-token leakage with a still-plausible loss curve. Upstream needs a
  **model capability flag** (e.g. `ModelSpec.cp_requires_contiguous_shards`)
  rather than a name test.
- **Bypassing `attention_masks` CP sharding**: causal-only models never consume
  the masks, and upstream's BlockMask CP sharding requires
  `seq_len % (cp * 128) == 0`, which SFT batches do not guarantee. Same
  argument: this wants a capability flag.

Filing these two needs an issue proposing the flag first. Recommended order:
file this PR (3 generic fixes) → open an issue for the capability flag → then a
follow-up PR.

## Notes for the filer

- Drop the `[kimi_k3]` prefix from the fork commit message; defects 1-3 are
  model-agnostic.
- Split from PR17 (checkpoint interval) deliberately — unrelated bug.
