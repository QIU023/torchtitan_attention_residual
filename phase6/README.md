# Phase 6 — Pre-merge infra completeness for upstream AttnRes PR

## Framing change vs Phase 5

**Phase 5** validated AttnRes itself (LM-only quality in Phase 4 + multimodal
quality in Arm 1 + PP cache adapter cross-modality invariance in Arm 2). The
goal there was *"AttnRes works"*.

**Phase 6** is a different goal. The torchtitan maintainer's stated position
is:

> "I don't want to see AttnRes stitched onto random other models with no
> large-scale pretrain validation. The PR can wait — when Kimi's next-gen
> model with AttnRes lands publicly, that becomes the merge trigger."

So our remaining work is **not** "validate AttnRes on more architectures /
more data" (that's a research arm Kimi will do internally). It is:

> *"By the time Kimi-AttnRes-NextGen drops, torchtitan must already have
> every infra hole filled so the merge is a one-line model registration."*

That re-orders the priority list. Quality numbers, scaling-law sweeps, and
non-Kimi backbones are out of scope. Infra completeness is everything.

## What "infra holes" means concretely

Three categories, in priority order:

### Category A — PP / parallelism gaps not covered by Phase 3-5

Phase 3 validated PP=4 V=2 (text-only). Phase 5 Arm 2 validated PP=4 V=2
(multimodal, fresh-init + C4 codepath at commit 54bb2dd, max |Δ|=0.013 nats).
These are the only matrices verified. Real Kimi infra at scale will use
deeper PP, TP+PP, and async checkpoint.

| # | Item | Why upstream cares |
|---|---|---|
| A1 | **Arm 2 on real LLaVA-Pretrain + Phase 4 ckpt** (the one Phase 5 Arm 2 always promised but blocked on Arm 1 reference curve) | Maintainer will ask: "fresh-init alignment is nice, but production-realistic init?" Need this answered before PR review. |
| A2 | **PP=8 V=4** alignment matrix | Kimi-NextGen at 100B+ scale will use PP > 4. Cache adapter math is depth-agnostic but P2P shape stability + Interleaved1F1B lookahead under V=4 needs explicit smoke. |
| A3 | **TP + PP + AttnRes** three-axis parallelism | Phase 5 launchers all use TP=1. AttnResProjection's (RMSNorm + linear) needs ColwiseParallel/RowwiseParallel registration in `parallelize.py`. Phase 4's `parallelize_kimi_linear` doesn't TP-wrap AttnRes layers. |
| A4 | **Async DCP checkpoint** with AttnRes state | `--checkpoint.enable_async` not validated for AttnRes (pseudo-query weights). Sync save costs ~30s every save_freq, won't scale to multinode wallclock. |
| A5 | **Resume from interrupted mid-save ckpt** | DCP partial-write recovery on AttnRes state has no test coverage. SIGTERM during save → restart loss-curve continuity test. |
| A6 | **Cross-parallelism numerical determinism smoke** | Standard PR-review question: same seed + data → bit-identical loss to bf16 epsilon across {FSDP-only, FSDP+PP, FSDP+PP+TP}? Without an explicit smoke we can't claim invariance under arbitrary parallelism composition. Run a small-config matrix (12 layers, 1B params) at 50 steps each and tabulate max\|Δ\|. |

### Category B — Multimodal-specific gaps

Kimi-NextGen is almost certainly multimodal. Phase 5's multimodal trainer is
the foundation, but it makes simplifying assumptions that don't survive
contact with realistic VLM data.

| # | Item | Why upstream cares |
|---|---|---|
| B1 | **Variable image count per row** (drop fixed `n_image_per_row == expected_per_row` assert) | LLaVA-Pretrain is 1-image-per-row. Real VLM data: zero-image (text-only mixed in), multi-image, video frames. The current assert in `multimodal_model.py:97-103` crashes the moment data is non-uniform. |
| B2 | **Image-text interleave** (image tokens not restricted to prefix) | LLaVA-Pretrain layout is `[<img>×196] [BOS] [caption]` — vision strictly at the start. InternVL / DeepSeek-VL2 / Kimi-VL 1.5 are interleaved (image at any position). Need: (a) `multimodal_dataset.py` collate handles arbitrary scatter positions, (b) PP cache adapter still preserves loss invariance when image_mask is non-contiguous. |
| B3 | **Vision tower FSDP-shard** (not replicated) — *deferred to "stretch"* | SigLIP-Base 92M / SO400M 400M / InternViT 6B all fit replicated on 4×5090 32G (≤4 GB / rank). FSDP-sharding the vision tower only matters when the vision encoder grows past per-rank capacity, which is a multinode + giant-encoder scenario we can't validate on current hardware. Spec the API surface (a single `wrap_vision_tower(parallel_dims)` entry) but defer implementation until an actual giant vision encoder lands. |
| B4 | **Tokenizer-aware sentinel selection** | Current `IMAGE_TOKEN_ID=32000` is "utility" in Llama-3.1's BPE — collision risk if caption legitimately contains "utility" tokens. Need per-tokenizer sentinel registry + startup assertion. |
| B5 | **AttnRes inference kv-cache support** | Generation-time question: does inference need to cache all N+1 block outputs, or only the final aggregated state? AttnRes shifts the answer from standard KV-cache. Maintainer will ask this for the inference path. Settle the answer + add a generation smoke (50 token autoregressive decode) that confirms the chosen scheme matches training-time logits. |

### Category C — PR-review polish

| # | Item | Why upstream cares |
|---|---|---|
| C1 | **Cache adapter ablation table** (bytes saved / throughput / loss diff distribution) | Maintainer will ask "quantify the value." Need empirical bytes_saved vs L,N,B,T,D + matched-step loss histogram. |
| C2 | **CPU pytest matrix expansion** | Phase 5 has 4 unit tests. Expand to: dynamic shape inference, mixed dtype, state_dict round-trip, partial failure recovery. |
| C3 | **Doc rewrite** (`attn_res/README.md` + `phase5/README.md` → architecture diagram + verified matrix + known limitations) | Direct paste into PR description. |
| C4 | **Performance regression CI** | Lock in current-baseline throughput (tps, peak memory, MFU) on a small config; CI runs 50-step smoke per PR and fails if any metric regresses >5%. Catches accidental perf hits introduced by future refactors. |

### Out of scope (explicitly)

- Putting AttnRes on Qwen2 / Llama3 / DSv3 base models. Maintainer
  doesn't want this.
- Scaling-law sweep 194M → 528M reproduction. Kimi will publish their
  own numbers.
- Caption / VQA quality benchmarks. Same reason.
- HF weight loader for current Kimi-Linear-48B-A3B-Base. Different model
  family from NextGen-AttnRes; loader for the eventual release model is
  the relevant one but it doesn't exist yet.

## Plan (3-4 weeks, single 4×5090 box)

| Week | Track A (parallelism) | Track B (multimodal) | Track C (polish) |
|---|---|---|---|
| W0 | Finish Phase 5 Arm 1 → A1 (Arm 2 real-data alignment) | — | C2 (start tests, run continuously) |
| W1 | A2 (PP=8 V=4), A3 (TP+PP smoke) + A6 (determinism matrix) | B1 (variable image count) | C2 grows with each A/B item |
| W2 | A4 (async DCP), A5 (mid-save resume) | B2 (interleave), B4 (sentinel registry) | C1, C3 |
| W3 (stretch) | — | B3 spec only (no impl) + B5 (kv-cache for AttnRes inference) | C4 (perf regression CI) |

C2 (CPU pytest) **runs continuously, not at the end** — every A/B item
lands with its own unit tests in the same PR. C2 in W2 is the final
matrix-completeness pass.

B3 is **spec'd but deferred** — vision-tower FSDP-shard is premature
optimization until we have a >4-GB-per-rank frozen vision encoder to
validate against, which our 4×5090 box cannot host.

Each Track A/B item ends with: launcher + test + alignment plot + 1-paragraph
writeup. Track C consolidates into PR-ready form.

## Concrete first-week actions (right now)

1. Let Phase 5 Arm 1 run to step 5000-6000 (caption loss target ≤ 2.8). That
   produces the reference curve A1 needs.
2. As soon as Arm 1 has 5k+ steps logged, launch Arm 2 with
   `INIT=weak_ckpt INIT_CKPT=phase4/runs/.../step-8000` matched seed +
   same data shuffle as Arm 1 → 2k step alignment → loss diff plot at
   matched steps.
3. In parallel, on this same box (CPU work, doesn't touch GPU): start
   B1 — write `MultiImageDataset` that emits variable N_vision per row
   + write the test that exercises the LM forward path with mixed
   (1-image, 0-image, 2-image) microbatches.

## Success criteria for the eventual upstream PR

When Kimi-AttnRes-NextGen drops, the upstream PR should be able to claim:

- AttnRes math validated on KDA+MLA+MoE backbone (Phase 4)
- Cache adapter loss invariance under PP×V×(text|multimodal)×(fresh|trained)
  init combinations — full matrix from Phase 3 + Phase 5 + Phase 6 A1/A2
- TP+PP+AttnRes interop verified (Phase 6 A3)
- Cross-parallelism numerical determinism (Phase 6 A6)
- Async DCP + AttnRes state safe (Phase 6 A4/A5)
- Multimodal trainer handles real VLM data layout
  (variable image count, interleave) (Phase 6 B1/B2)
- AttnRes-aware inference kv-cache scheme (Phase 6 B5)
- Vision tower FSDP-shard API spec'd for >1B vision encoders (B3,
  impl deferred until hardware allows validation)
- ≥20 CPU-runnable unit tests + 5 GPU smokes in CI
- Performance regression guard in CI (Phase 6 C4)
- Architecture doc + verified-config matrix + known-limitation list

If all of that's green, the PR is "ready for the next-gen model" and the
maintainer's blocker (no-large-scale-validation-on-stitched-models)
becomes moot — the model itself when it ships brings the validation.

## Risk: NextGen-Kimi shape change

Phase 6's "1-line registration" claim only holds if Kimi-NextGen keeps
the current `KimiLinearAttnResModel` shape (KDA + MLA + MoE backbone,
shared `AttnResProjection(d → 1, no bias)`, vocab 163840, etc).
Plausible deviations that would break the claim:

- KDA → standard MHA/GQA → `parallelize_kimi_linear`'s KDA-specific
  TP wrap doesn't apply
- `AttnResProjection` from `Linear(d → 1)` to `Linear(d → k)` (multi-head
  pseudo-query) → scatter / softmax-pool path changes
- Vocab change → embed/lm_head TP wrap shape
- New layer types (state-space, mamba-hybrid) interleaved with KDA/MLA

Mitigation: write Phase 6 infra at the **API level**, not hardcoded to
current shape. Example: `wrap_attn_res_projection(module, in_dim,
out_dim, tp_dim=None)` parameterizes both dims rather than assuming
`out_features=1`. This costs minimal extra code now and absorbs most
plausible NextGen shape changes without re-implementation.
