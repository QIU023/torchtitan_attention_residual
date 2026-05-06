# Phase 10 Stage J — Autoregressive Inference Fabric

Captures the fabric pattern of generation-phase inference (vs the
batch-forward fabric in Stage D). Without a real KV cache (porting
KDA's recurrent state through the fla-core Triton kernel boundary
is non-trivial — the autotuner pre-hook expects all kwargs non-None
and seq_len=1 trips it), we run the **growing-prefix** mode: each
new token re-forwards the full prefix.

## Run config

* Mesh: same as Stage D (FSDP=4 × TP=2 × EP=2 = 8)
* Generations: 20 independent prompt batches
* Tokens per generation: 20
* Initial prompt length: 64 tokens
* Total forward calls: 400
* Wall time: 67.3 s

## Captured fabric (growing mode)

```
96000 × Send         128 B   nranks=2  (EP all-to-all dispatch — per token per MoE layer)
96000 × Recv         128 B   nranks=2  (EP all-to-all combine)
48000 × AllGather   29.6 MB  nranks=2  (FSDP unshard, ring-algorithm sub-events)
51200 × AllGather   1-16 MB  nranks=4  (FSDP per-layer reconstitution)
~30000 × AllReduce  varying  nranks=2  (TP attention output AR; size grows with prefix length)
```

The `AllReduce` size rows are particularly informative — the trace
shows AR sizes climbing as prefix grows: 350KB, 355KB, ..., 388KB.
This is the diagnostic signature of "no-cache autoregressive" — every
re-forward emits a TP collective sized proportional to the current
prefix length. With a real cache the per-token TP collective would
stay constant-tiny.

## Pattern signature (compared to other regimes)

| Regime | Per-call TP AR size | Per-call EP A2A size | Calls per token |
|---|---|---|---|
| Stage D batch fwd (seq=512) | 12 MB | ~24 MB | 1 (whole batch) |
| Stage J growing (seq grows P+i) | 350 KB → 388 KB | ~128 B → growing | many (per layer per token) |
| Stage J ideal-cache (seq=1, blocked) | constant ~4 KB | constant ~128 B | many |

The "ideal-cache" mode is **blocked by KDA Triton kernel**: the
fla-core autotuner pre-hook calls ``kwargs[name].clone()`` over
restore-value names; at seq_len=1 some kernel kwargs are None and
the call crashes with ``AttributeError: 'NoneType' object has no
attribute 'clone'``. Documented; impl pending.

## Implications for IXIA modeling

Inference deployments will exhibit a mix:
1. **Prefill phase**: Stage D fabric (single large batch fwd)
2. **Decode phase**: Stage J ideal-cache fabric (many small per-token
   calls), or Stage J growing-prefix fabric if no cache

Block AttnRes adds to both regimes equally (the AttnRes aggregation
is local; only the AttnRes pseudo-query / norm parameters add a tiny
amount to the FSDP unshard volume).

## Files

* `phase10/inference_autoregressive.py` — generation loop with
  ``--mode {growing,single_token}`` switch.
* `phase10/run_autoregressive.sh` — runs both modes with tier_b
  NCCL trace + pipeline → ixia_config.json.
* `phase5/runs/inference_autoregressive_growing/tier_b_trace/` —
  growing-prefix mode trace artifacts.
* `phase5/runs/inference_autoregressive_single_token/run.log` —
  single_token mode log showing the KDA Triton kernel blocker.

## Future: real KV cache port (~1-2 days)

To make ``single_token`` mode work and obtain the constant-msg-size
ideal-cache fabric, the path is:
1. Port HF reference's ``KimiDynamicCache`` pattern (~200 LOC) to
   our ``torchtitan.experiments.kimi_linear.model``.
2. Modify ``KimiMLAAttention.forward`` to accept ``cache`` param and
   append ``key_states`` / ``value_states`` to it. Standard.
3. KDA is harder: needs the recurrent state ``h_state`` written
   after each ``chunk_kda`` call and re-fed in. Triton kernel needs
   updated signature.
4. Wire the cache through ``KimiLinearAttnResModel.forward`` and
   the AttnRes layer's wrapper.

Out of Phase 10's 21h budget; tagged as Phase 11 future work.
