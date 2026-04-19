# Phase 0 Analysis: AttnRes in torchtitan — Go/No-Go Decision

**Date**: 2026-04-12
**Status**: **GO for torchtitan**

---

## 1. Executive Summary

torchtitan's PP infrastructure supports Block AttnRes integration **without modifying PyTorch's pipelining library**. The implementation is model-level only: modify `Decoder.forward()` to thread block summaries between layers and return tuple outputs for inter-stage communication. `PipelineStage` natively handles tuple tensor outputs via P2P send/recv.

---

## 2. Key Findings

### 2.1 Moonshot Reference Repo
- **No runnable code** — only PDF + pseudocode in README
- Pseudocode is clear and sufficient for implementation (Figure 2 in paper)
- Important detail: `block_size` counts ATTN + MLP sublayers; each transformer layer has 2

### 2.2 PyTorch Pipelining API (stage.py)

**Critical discovery**: `PipelineStage` supports tuple outputs natively.

```python
# stage.py:31 — _normalize_model_output_as_tuple
output_tuple = output if type(output) is tuple else (output,)

# stage.py:433 — get_fwd_send_ops iterates over tuple elements
for idx, out in enumerate(output_tuple):
    dst_stages = self.act_send_info[idx]
    for dst in dst_stages:
        ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))
```

**Activation flow**:
1. `forward_one_chunk()` (line 660): calls `_retrieve_recv_activations()` for non-first stages
2. Received activations become **positional args** to `submod.forward()`
3. If stage N-1 returns `(tensor_a, tensor_b)`, stage N's forward is called as `forward(tensor_a, tensor_b, **kwargs)`

**No hook mechanism** — no `pre_send`/`post_recv` callbacks. But tuple I/O makes hooks unnecessary for the basic approach.

### 2.3 torchtitan Model Architecture

**Decoder.forward()** (`models/common/decoder.py:124`):
```python
def forward(self, tokens, attention_masks=None, positions=None):
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
    for layer in self.layers.values():
        h = layer(h, self.freqs_cis, attention_masks, positions)
    h = self.norm(h) if self.norm is not None else h
    output = self.output(h) if self.output is not None else h
    return output
```

**PP stage handling**: `pipeline_module_split()` deep-copies the model and prunes layers. Intermediate stages have `tok_embeddings=None`, `norm=None`, `output=None`. The `if ... is not None` guards handle passthrough.

**kwargs propagation**: `attention_masks` and `positions` are passed as kwargs to ALL stages via `pp_schedule.step(**extra_inputs)` in `trainer.py:670`.

### 2.4 Kimi Infra Engineer Implementation Notes (Reku, Zhihu)

Key quotes (translated):

> "The general approach is to add an adapter after pipeline parallel communication, which concatenates the received block with cached blocks in the adapter. The backward is similar — receives grads for all blocks, accumulates them in the adapter, and sends the accumulated buffer to the next stage when needed. The code logic is quite symmetric and doesn't affect the network's internal logic."

> "Under interleaved pipeline scheduling, send/recv overhead is easily hidden in steady state, only warmup and cooldown expose a bit of communication."

> "Cross-stage caching changes the accumulation order [of gradients], causing debug/precision alignment difficulties when PP config changes."

> "They (Guangyu/Zhang) designed block attention residual almost overnight — its locality makes cross-stage caching communication optimization straightforward."

---

## 3. Implementation Architecture

### 3.1 Model-Level Changes (Phase 2: Single-Card Correctness)

**Modified forward signature**:
```python
class Decoder(BaseModel):
    def forward(self, tokens, blocks=None, *, attention_masks=None, positions=None):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        if self.use_attn_res:
            if blocks is None:
                blocks = [h]  # token embedding = first "block"
            partial_block = h

            for layer in self.layers.values():
                blocks, partial_block = layer.forward_attn_res(
                    blocks, partial_block, self.freqs_cis, attention_masks, positions
                )

            # Final cross-block attention for output
            h = block_attn_res(blocks, partial_block, self.final_attn_res_proj, self.final_attn_res_norm)
        else:
            for layer in self.layers.values():
                h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h

        # For PP intermediate stages: also return blocks
        if self.use_attn_res and self.output is None:
            blocks_tensor = torch.stack(blocks)  # [N, B, T, D]
            return (output, blocks_tensor)
        return output
```

**Modified TransformerBlock** (following paper pseudocode):
```python
class Llama3TransformerBlock(TransformerBlock):
    def forward_attn_res(self, blocks, partial_block, freqs_cis, attention_masks, positions=None):
        # Apply block attnres before attention
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        # Block boundary check
        if self.layer_id % self.block_size == 0:
            blocks.append(partial_block)
            partial_block = None

        # Self-attention
        attn_out = self.attention(self.attention_norm(h), freqs_cis, attention_masks, positions)
        partial_block = partial_block + attn_out if partial_block is not None else attn_out

        # Apply block attnres before MLP
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

        # MLP
        mlp_out = self.feed_forward(self.ffn_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block
```

**New parameters per layer** (negligible overhead):
- `attn_res_proj`: `nn.Linear(d_model, 1, bias=False)` — pseudo-query (MUST be zero-initialized)
- `mlp_res_proj`: `nn.Linear(d_model, 1, bias=False)` — pseudo-query (MUST be zero-initialized)
- `attn_res_norm`: `RMSNorm(d_model)`
- `mlp_res_norm`: `RMSNorm(d_model)`

### 3.2 PP Integration (Phase 3: Naive Approach — No Caching)

**How it works with existing PipelineStage**:

1. **Stage 0** (first): `forward(tokens, blocks=None, attention_masks=..., positions=...)`
   - Creates `blocks = [tok_embedding]`
   - Processes layers 0..K-1, accumulates block summaries
   - Returns `(partial_block, blocks_tensor)` — tuple of 2 tensors

2. **Stage 1..N-2** (middle): `forward(partial_block, blocks_tensor, attention_masks=..., positions=...)`
   - Positional arg mapping: `tokens=partial_block`, `blocks=blocks_tensor`
   - Unstacks blocks, processes layers, returns updated `(partial_block, blocks_tensor)`

3. **Stage N-1** (last): Same as middle but applies final norm + output
   - Returns single `logits` tensor (not tuple)

**Communication cost per transition**: O(N * B * T * D) — sends all block summaries. This is the naive approach without caching.

### 3.3 Cross-Stage Caching Optimization (Phase 3+: Advanced)

The Kimi engineer's adapter pattern:
- Each physical stage caches blocks received during previous virtual stages
- After first VP chunk completes, all blocks are distributed
- Subsequent VP chunks only send incremental blocks (O(P * Np * d) per transition vs O(C * Np * d))

**Implementation options in torchtitan**:
- **Option A**: Subclass `PipelineStage` — customize `get_fwd_send_ops`/`get_fwd_recv_ops`
- **Option B**: Model-internal caching — Decoder maintains a `_blocks_cache` dict keyed by (microbatch_id, virtual_stage_id)
- **Option C**: Propose upstream hook API to PyTorch pipelining (longer-term)

**Recommendation**: Start with naive approach (Phase 3), add caching as Phase 3+ optimization. The naive approach is correct and demonstrates the core algorithm. Caching is a performance optimization that can be profiled and added incrementally.

---

## 4. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Variable-size blocks tensor across stages | Low | PipelineStage infers metadata per-stage; different stages can have different output shapes |
| Forward signature change breaks non-PP path | Low | `blocks=None` default; `use_attn_res=False` preserves original behavior |
| Gradient correctness through stacked blocks | Medium | Extensive unit tests; compare single-card vs fake-PP numerical outputs |
| torchtitan maintainers reject model-level approach | Medium | Follow CLAUDE.md guidelines; start as experiment, propose to core if accepted |
| blocks_tensor consumes too much activation memory | Low | O(N*d) per token where N~8; negligible vs per-layer activations |
| Pseudo-query zero-init not preserved through model converters | Medium | Add explicit zero-init check in `init_states()` |

---

## 5. File Inventory — What to Modify

### Core changes:
- `torchtitan/models/common/decoder.py` — Decoder.forward() signature + AttnRes loop
- `torchtitan/models/llama3/model.py` — Llama3TransformerBlock with AttnRes parameters
- `torchtitan/models/llama3/config_registry.py` — Add AttnRes config options

### New files:
- `torchtitan/models/common/attn_res.py` — `block_attn_res()` function + `AttnResConfig`

### Config changes:
- `torchtitan/config/` — Add `--model.use_attn_res`, `--model.attn_res_num_blocks` flags

### Tests:
- `tests/unit/models/test_attn_res.py` — Unit tests for Block AttnRes correctness
- `tests/integration_tests/` — PP integration test with AttnRes

---

## 6. Estimated Effort

| Phase | Duration | Compute | Cost |
|-------|----------|---------|------|
| Phase 2: Single-card correctness | 5-7 days | 1x 5090 | ~$40 |
| Phase 3: Naive PP (fake PP + real 8-card) | 7-10 days | 1x 5090 + 8x 5090 | ~$280 |
| Phase 3+: Cross-stage caching | 3-5 days | 8x 5090 | ~$130 |
| Phase 5: PR + Blog | 3-5 days | — | — |
| **Total** | **~4 weeks** | | **~$450** |

> Price baseline (Vast.ai, live quotes 2026-04-13):
>
> - **1x RTX 5090** (California, AMD EPYC 7742, PCIE 4.0 x16): **$0.362/hr**, DLPerf 193.7, verified, reliability 95.4%
> - **8x RTX 5090** (Oregon, AMD EPYC 7K62, PCIE 4.0 x16, 297 ports): **$2.616/hr**, DLPerf 475.5, verified, reliability 98.5%
>
> Assumptions: debug-heavy phases budgeted at ~50% wallclock GPU utilization; training phases billed continuous.

---

## 7. Next Steps

1. **Open RFC issue on torchtitan** — describe the approach, link paper, get maintainer feedback
2. **Start Phase 2** — implement `block_attn_res()` and single-card correctness tests
3. **Rent 1x 5090** (Vast.ai, ~$0.362/hr) — validate loss curve alignment with paper's small-scale results
