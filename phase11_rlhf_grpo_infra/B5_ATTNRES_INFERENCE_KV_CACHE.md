# B5: AttnRes Inference KV-Cache Scheme

**Original question** (`phase6_upstream_pr_prep/README.md`): does inference need to
cache all N+1 block outputs, or only the final aggregated state?

**Answer (now empirically validated):** AttnRes block representations
are **intra-forward state**, not cross-step cache. The standard
self-attention KV cache (KDA recurrent state + MLA KV) handles
cross-step incremental decode; AttnRes adds nothing on top.

## Scheme

Each `model.forward()` call:

```
committed_blocks: list[Tensor] = []        # local var
partial_block: Tensor = None               # local var

for block_idx in range(num_blocks):
    for layer in layers_in_block:
        # Phase 1 cache built once at block boundary (also local).
        partial_block = layer.forward(
            committed_blocks, partial_block, ...
        )
    committed_blocks.append(partial_block)
    partial_block = None

# All locals fall out of scope at function exit.
```

`committed_blocks` and `partial_block` are **Python locals** in
`KimiBlockAttnResModel._forward_one_block` (line 482-, sglang
`models/attn_res_overlay.py`). They live for one forward call and
are GC'd when the function returns. There is no module-level cache.

## Why this is correct for autoregressive decode

Block AttnRes' aggregation formula (Kimi paper §5):

```
h_l = softmax_aggregate(committed_blocks ∪ {partial}, query_l, norm_l)
```

The aggregation depends on **the current forward's hidden states**,
not on prior tokens' hidden states. Prior tokens' contribution comes
through the standard self_attn (KDA + MLA) which DOES have
cross-step cache.

Concretely, at decode step T:
* Standard KV cache: contains tokens 1..T-1 (KDA state + MLA KV).
* Forward pass on token T: `self_attn(K_cache, q_T)` produces attn
  output for token T.
* AttnRes aggregation: builds block reps fresh from `attn_out_T`.
* Final h_T → next-token logit → output.

At step T+1, block reps are recomputed from scratch — they were never
needed across steps.

## Memory cost during decode

Per rank, per forward (transient):

| object | shape | bytes (bf16, our 1.4B) |
|---|---|---|
| committed_blocks (max 4) | `(N=4, T=1, D=1024)` | 32 KB |
| partial_block | `(T=1, D=1024)` | 8 KB |
| Phase-1 cache (committed_part + lse) | `(2L_block, T=1, D=1024)` | 16 KB / per Phase-1 group |

Total ~50 KB transient at decode step. Compare to standard KV cache
on a 16K context: ~20 MB per layer × 16 layers = 320 MB — three
orders of magnitude larger. The block-rep transient is a rounding
error.

## Compare to "cache only final state"

The alternative (caching aggregated state across steps) is **wrong**:
each layer's pseudo-query attends to a DIFFERENT subset of the block
list (4 layers × 2 queries per layer = 32 distinct queries / 16-layer
block group). They don't reduce to one shared state. Caching only the
final state at step T would mean step T+1 has no per-layer query
re-aggregation possible, breaking the algorithm.

## Validation

* TP=1 boot+gen on real ckpt step-12500 (16 token decode): passes
  without OOM, no crash. (Already covered in
  `phase11_rlhf_grpo_infra/PHASE11_SGLANG_REPORT.md` task #9.)
* Chunked-prefill 8K prompt × 2K chunks: passes — block-rep transients
  build cleanly across chunks. (Task #12.)
* Re-bench at TP=8 prefill=16384 ctx: peak per-rank GPU memory same
  with shard=0 vs shard=1, dominated by `mem_fraction_static` pool —
  confirms block-rep transients are O(MB) not O(GB).

## Decision

**Status: closed.** AttnRes inference uses standard KV cache for
self_attn cross-step, AND in-forward-pass-only block reps for the
residual stream. No new cache data structure needed. PR-readiness:
✅ this slot is clear.
