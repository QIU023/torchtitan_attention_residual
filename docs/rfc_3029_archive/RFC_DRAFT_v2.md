# [RFC] Block Attention Residuals (Kimi, 2026): single-GPU primitive + cross-stage caching adapter

## Summary

Implement **Block Attention Residuals** (AttnRes) from [Kimi Team 2026, *Attention Residuals*](https://arxiv.org/abs/2603.15031) in torchtitan. The submission is split into two PRs: (1) a self-contained `experiments/attn_res/` that adds the primitive and an end-to-end FSDP-compatible Llama3 variant with single-GPU loss-curve evidence (this PR, ready now); (2) a follow-up that adds the **cross-stage caching adapter** that hides AttnRes's inter-block communication inside interleaved 1F1B steady-state on PP + VP — the engineering value-add and the headline claim.

## Motivation

Standard residuals `h_{l+1} = h_l + f_l(h_l)` treat every layer as equal-weight; hidden-state magnitude grows with depth and shallow-layer signal is diluted. AttnRes replaces the fixed add with softmax attention over preceding layer outputs, using a per-layer learned pseudo-query vector. The paper reports **AttnRes ≈ baseline × 1.25 compute at matched model size**, with block boundaries making the method PP-compatible at `O(N d)` extra memory instead of `O(L d)`.

Paper artifacts and background:
- arXiv: https://arxiv.org/abs/2603.15031 · reference implementation: https://github.com/MoonshotAI/Attention-Residuals
- Kimi infra engineer's implementation notes (Megatron pipeline adapter sketch): https://www.zhihu.com/question/2016993095078684011

The Kimi Team shipped this inside a proprietary fork of Megatron-LM. No open-source framework has integrated it yet. torchtitan is a strong fit because the block boundary aligns with PP stage boundary, and Block AttnRes's `O(N d)` cross-stage traffic is exactly the regime where torchtitan's PP + VP schedule shines.

## Algorithm (one paragraph)

Partition `L` layers into `N` contiguous blocks (sweet spot `N ≈ 8`). Inside a block, use standard residuals. At each sub-layer, read the input via softmax attention over all previous committed block representations plus the current partial sum:

```
V = stack(blocks + [partial_block])             # [N+1, B, T, D]
K = RMSNorm(V)
w = block_attn_res_proj.weight                  # learnable pseudo-query, [D]
logits = einsum("d,nbtd->nbt", w, K)
weights = softmax(logits, dim=0)
h = einsum("nbt,nbtd->btd", weights, V)
```

At a block boundary, commit the current partial and start a fresh one. **Pseudo-queries are zero-initialized** so initial softmax weights are uniform, making step 0 numerically equivalent to standard residuals. See paper Figure 2 for pseudocode.

## Scope: two PRs

- **PR #1 (this, ready)**: `experiments/attn_res/`. Self-contained primitive + `AttnResLlama3Model` (subclass of `Llama3Model`) + Llama3-150M Trainer configs + 14 CPU unit tests + single-GPU FSDP loss-curve evidence. No core torchtitan files modified.
- **PR #2 (follow-up, Phase 3–4)**: cross-stage caching adapter around `torch.distributed.pipelining._PipelineStage`, validated first on single-GPU fake-PG `PP=4`, then on 8× RTX 5090 PCIe with `PP=8, VP=2` interleaved 1F1B on a 1.5–2B dense Llama3. Headline claim: **Block AttnRes keeps PP overhead < 5% even over PCIe bandwidth** — a stronger statement than the paper's NVLink setting.

## Placement decision

Per `torchtitan/experiments/README.md` principles 3 & 5 and `.claude/CLAUDE.md` core-principle #4, the experiment does not touch core. Code layout:

```
torchtitan/experiments/attn_res/
├── README.md
├── __init__.py            # model flavors: debugmodel_attn_res, 150M_attn_res; model_registry()
├── attn_res.py            # block_attn_res primitive, AttnResProjection, stack/unstack helpers
├── model.py               # AttnResLlama3TransformerBlock(Llama3TransformerBlock), AttnResLlama3Model(Llama3Model)
├── config_registry.py     # Trainer configs: llama3_150m_baseline, llama3_150m_attn_res
└── tests/
    └── test_attn_res.py   # 14 CPU unit tests
```

Plus one line in `torchtitan/experiments/__init__.py` registering `"attn_res"` in `_supported_experiments`. Usage: `--module attn_res --config llama3_150m_attn_res`. The subclass pattern overrides `forward` with an `is_block_start` kwarg that dispatches to `forward_attn_res`, so FSDP's pre-forward `all_gather` hook fires and AttnRes sub-params unshard before `rms_norm` — this is required; calling `forward_attn_res` directly bypasses FSDP and breaks under DTensor.

## Phase 2 evidence (this PR)

Setup: RTX 5090 32 GB, torch 2.11 cu130, Llama3-150M dense (dim=768, 12 layers, 6 blocks, GQA n_kv_heads=4, tied embeddings, vocab=128256), BF16 FSDP, C4-en streaming, Llama-3.1 tokenizer, `seq_len=2048`, `local_batch_size=8` × `grad_accum=2` = global 16, lr 3e-4 cosine, warmup 500, decay to 10% over 80% of steps, 20k steps = ~655M tokens.

Identical config across both runs; only `model_spec` differs.

Same-step train-loss milestones (from TensorBoard, ±50-step window):

| step | baseline | attn_res | Δ (attn_res − baseline) |
|---:|---:|---:|---:|
| 500   | 6.1412 | 6.0146 | −0.1265 |
| 1000  | 5.3967 | 5.3069 | −0.0898 |
| 2500  | 4.7457 | 4.6362 | −0.1095 |
| 5000  | 4.3575 | 4.2696 | −0.0879 |
| 7500  | 4.2667 | 4.1763 | −0.0904 |
| 10000 | 4.3235 | 4.2192 | −0.1043 |
| 12500 | 3.9395 | 3.8697 | −0.0699 |
| 15000 | 3.7368 | 3.6861 | −0.0507 |

The `Δ` stays negative at every milestone, with the magnitude narrowing as both runs approach cosine end (expected: AttnRes's "equivalent 1.25× compute" becomes harder to distinguish when both runs are near their loss floor at this size). Full 20k step comparison plot will be attached; scripts in `phase2_attnres_baseline_loss/` reproduce end-to-end.

Throughput / memory (single GPU FSDP, no PP, no torch.compile):

| metric | baseline | attn_res | delta |
|---|---:|---:|---|
| tps | ~71 k | ~50 k | −30 % |
| peak memory | 29.1 GiB | 30.1 GiB | +1.0 GiB (matches O(Nd) with N=6, d=768) |

The 30 % tps gap is expected: every sub-layer now does a stack + RMSNorm + einsum + softmax + weighted sum over `N+1` block activations. The gap is expected to collapse substantially under PP (activations don't need to be re-stacked per stage) and under `torch.compile` — which is explicitly out of scope for this PR.

Correctness invariants covered by 14 CPU unit tests:
- `block_attn_res` with zero pseudo-query = uniform average (step-0 equivalence to standard residuals)
- Softmax weights sum to 1 per token position
- Gradient flows to blocks, partial, pseudo-query, norm weight
- `stack_blocks` / `unstack_blocks` round-trip preserves autograd
- End-to-end `AttnResLlama3Model` forward + backward; pseudo-queries are exactly zero after `init_states`
- PP intermediate-stage simulation (strip `tok_embeddings`, `output`, `norm`) returns `(partial_block, stack_blocks(blocks))` tuple so `PipelineStage` P2P sends both tensors

Known fix captured in this PR: `Llama3TransformerBlock.forward_attn_res` called directly bypasses FSDP's pre-forward `all_gather` hook, leaving AttnRes sub-params as `DTensor` and breaking `rms_norm`. The subclass `forward` dispatches AttnRes kwargs via `__call__` so the hook fires.

## Phase 3–4 plan (PR #2 preview)

Per a survey of torchtitan's PP surface, `torch.distributed.pipelining`'s P2P is schedule-internal; there are no op-level send/recv hooks we can control without forking PyTorch. The tractable insertion point is **wrapping the stage submodule** via `ModelSpec.pipelining_fn` — the same integration path `experiments/transformers_modeling_backend` uses. Our custom `pipeline_llm_with_cache_adapter` calls core `pipeline_llm` unchanged, then walks `schedule._stages` and wraps each `stage.submod` with `CrossStageCacheAdapter`. **Zero modifications to core torchtitan.**

The adapter implements Kimi's "把收到的 block 与适配器中缓存的 block 进行拼接" pattern at the Python level:

- **Forward**: on entering stage `N+1`, receive `(partial_block, incoming_blocks)` from P2P, **concat** `incoming_blocks` with the adapter's cached blocks from earlier microbatches, run the wrapped model, push the resulting blocks into the cache keyed by microbatch id.
- **Backward**: register autograd hooks on output tensors to **accumulate per-block grads** across microbatches; when the stage sends backward, send `partial_block.grad` plus the accumulated per-block grad buffers.

The adapter is env-flagged (`TORCHTITAN_ATTNRES_CACHE=1`) so naive full-stack PP and adapter-cached PP run from the same binary for A/B comparison.

Benchmark plan on `8 × RTX 5090 PCIe` (intentionally PCIe, not NVLink — the cheap/wide-deployment regime):

1. **Naive PP sanity** (500 steps, `PP=8, VP=2, FSDP inner, Llama3 150M`): confirm loss curve aligns with single-GPU Phase 2 reference, measure per-stage send size grow linearly in stage id. This catches most integration bugs in < 1 h.
2. **Adapter A/B** (same 500 steps, flag on): loss must match naive within bf16 tolerance; per-stage send size becomes constant.
3. **Scale-up headline run**: `Llama3 1.5-2B dense, 20B tokens, PP=8 VP=2 interleaved 1F1B`. Reported: step-time overhead (<5% target vs matched baseline), per-layer memory (paper predicts 5.5 d vs 3 d), MFU, NCCL comm trace showing AttnRes cross-stage send/recv hidden in steady-state.

## Open design questions (for maintainers)

1. **Adapter surface.** Is wrapping `stages[i].submod` with a `nn.Module` the canonical extension today, or is there a preferred hook we missed (e.g. a pre/post-`forward` registration we can attach to `_PipelineStage`)?
2. **Backward hooks under the interleaved schedule.** Does `PipelineScheduleMulti` preserve `torch.autograd.backward_hook` on intermediate activations across microbatches, or does its activation-recomputation interfere? We will test; opinions from @wconstab / @fegin would shortcut this.
3. **VP chunk ordering.** Block AttnRes's logical block index is monotonic with depth, but VP sends chunks out of depth order. How should the adapter key its cache — by `(microbatch_id, virtual_stage_id)` or by the chunk's logical block index?
4. **dtype policy.** The paper runs BF16 end-to-end including the AttnRes softmax. Is there a reason torchtitan's loss function upcasts to fp32 that we should mirror here (e.g. in the cross-block aggregation at the last stage)?
5. **Activation checkpointing.** `activation_checkpoint.mode=selective` interacts with FSDP2 reshard-after-forward. If the AC reruns `forward_attn_res`, the `all_gather` hook fires twice — benign but wasteful. Is there a lightweight way to opt the AttnRes sub-params into the AC-hidden unshard?

## Non-goals (for this RFC)

- **Inference.** The paper's two-phase computation (batched Phase 1 over block pseudo-queries + online softmax Phase 2) is out of scope. This RFC covers training only.
- **MoE.** No interaction with `deepseek_v3` / `gpt_oss` / other MoE model variants.
- **Other model families.** Only Llama3 is wired in PR #1. The primitive is model-agnostic; porting to qwen3 / llama4 is ~50 lines of subclass code but not in this RFC.
- **Core refactor.** We explicitly do not propose moving the subclass dispatch pattern back into core `Llama3TransformerBlock.forward`. If the experiment graduates, upstreaming is a separate discussion.

## Ownership

- Owner: @QIU023 — will maintain the experiment under `experiments/` rules (adapt to core changes, respond to issues, remove if it goes stale).
- Fork / branch: [`QIU023/torchtitan:attention_residual_dev`](https://github.com/QIU023/torchtitan/tree/attention_residual_dev), commit `144d10c` (squashed; prior commits amended during migration from core to experiments).
- Compute commitment for PR #2: ~$1k–1.5k on 8× RTX 5090 PCIe (vast.ai) to land the 8-GPU benchmark; single-GPU dev cost already absorbed.

---

**Signals that would accelerate this:** confirmation that experiments/ placement is the right starting point; a pointer on whether the cross-stage caching adapter should live under `distributed/` as a general utility or stay under `experiments/attn_res/` until it graduates; any prior art on autograd hooks across `PipelineScheduleMulti` microbatches we should study instead of starting from scratch.
