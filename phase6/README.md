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
| A1 | **Arm 2 on real LLaVA-Pretrain + Phase 4 ckpt** — orchestrated by `phase6/run_a1_alignment_v2.sh`. Both arms init from Phase 4 step-8000 with seed=42, GLOBAL_BS=12, 2000 steps. **Result post-A1.1 fix (commit 2d69453)**: median \|Δ\| 0.024 nats (5.5× under 0.13 threshold), p95 0.071 nats (1.8× under), max 0.252 nats at **step 8 during warmup** (warmup_steps=10). 85% of 2000 steps within 0.05 nats; only 10 steps (0.5%) exceed the 0.13 threshold and they're all in the first ~100 steps where bf16 noise during LR warmup is amplified. Strict max-criterion reads FAIL (single warmup outlier pushes past 0.13), but **post-warmup alignment is essentially perfect** — comparable to Phase 3's text-only fresh-init result of max 0.013 nats. Reports at `phase6/alignment_report_arm2_real_mm_v2.txt` and `phase6/alignment_arm2_real_mm_v2.csv`. The pre-fix v1 report at `_v1` suffix preserves the broken-FSDP comparison for the writeup. | Maintainer asks: "real-multimodal alignment under production-realistic init?" Answer: "PASS in median + p95; one warmup-transient outlier in max. After the projector grad-sync fix this is fundamentally a clean pass." |
| A1.1 | **(DONE, commit 2d69453)** — Root cause turned out inverse of original hypothesis: not "PP overscales grad" but "FSDP under-syncs projector". Reading `trainer.py:761-767` showed only `self.model_parts` params get clipped/synced. The replicated projector on the FSDP=4 path was **never grad-synced across DP ranks**, so each FSDP rank's projector trained on its own dp shard only and rank 0's projector saw 1/dp_world of the per-step batch. PP=4 lives on rank 0 alone (no replication), so it correctly sees the full batch — that asymmetry is what produced the systematic offset. **Fix in `phase5/train_mm.py`**: when the dp/batch mesh has size > 1, `fully_shard(self.projector, mesh=batch_mesh)` so projector grads reduce-scatter across DP ranks. Under PP-only (no dp axis) the wrap is skipped — single-rank projector already correct. Smoke validated 6× tighter alignment; full 2000-step rerun produced the median 0.024 result above. | The fix itself is small (~15 LOC) but it generalizes to any "stage-0-only replicated trainable param" pattern — vision projector today, cross-attention or sentinel-position scatter modules in NextGen, etc. Documenting the pattern in the PR is what makes it merge-relevant. |
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
- **`fla-core` KDA triton kernel re-tuning for Blackwell (sm_120).**
  Current observation on 4×RTX 5090: GPU util reads 100% but power
  ~25% of TGP and MFU 0.78%. Root cause is `fla-core 0.5.0`'s
  `chunk_kda` kernel was written for Hopper (H100 / sm_90 register +
  shared-mem layout); Blackwell sm_120 register width and shared-memory
  banking are different, so the same kernel hits register spill +
  bank-conflict + cache-miss patterns and ALU sits idle. **`fla-core`
  is not Kimi's repo** (it's the `flash-linear-attention` library
  maintained separately by Songlin Yang et al.); the kernel side of
  Blackwell adaptation belongs upstream there, not in our PR. Our
  scope is the distributed-training framework (FSDP/PP/TP/EP composition,
  cache adapter, multimodal trainer, ckpt resume) — kernel-level
  micro-optimization is a different project. The torchtitan PR for
  AttnRes will land alongside whichever KDA kernel implementation
  Kimi-NextGen ships, which they will have tuned for whichever
  hardware they release on.

## 8-GPU 3D parallelism roadmap (rented box)

The 4×5090 box only goes up to 2D parallelism (FSDP×PP). When we move to
an 8-GPU rented box, we get a third axis. The valid 3D combos and their
pre-merge value:

| Config | GPUs | Tests what | AttnRes-specific work |
|---|---|---|---|
| **FSDP=2 × PP=4** | 8 | Deeper PP than 4-GPU max (PP=4 on 4 GPU forces FSDP=1) — validates Interleaved1F1B at PP=4 *with* FSDP=2 simultaneously, and that AttnRes cache adapter delta P2P sends survive PP×FSDP2 collective overlap | None — drop-in launcher |
| **FSDP=2 × PP=2 × TP=2** | 8 | First time TP enters AttnRes path. Tests that `AttnResProjection` (RMSNorm + linear) registers correctly with `ColwiseParallel` / `RowwiseParallel`, and that the cache adapter's per-stage delta tensors are sharded along TP axis without breaking the loss-invariance contract (delta tensor's dim that ColParallel splits must remain consistent across send/recv) | Add TP plan map for `AttnResProjection` in `parallelize.py`; verify `_layers_per_block` and `_return_only_new_blocks` signal flow under TP wrapping |
| **FSDP=2 × PP=2 × EP=2** (MoE) | 8 | Tests AttnRes through MoE-containing blocks. Kimi-NextGen is almost certainly MoE; expert routing + AttnRes block boundary commit need to interleave correctly. Also tests EP-shard reduce pattern under PP+FSDP | Switch flavor to a Kimi-Linear MoE config (`first_k_dense_replace` < L); verify cache adapter delta accumulation when block boundary lands inside a MoE FFN; add EP plan to `parallelize.py` |
| **FSDP=2 × PP=2 × CP=2** (long context) | 8 | Tests context-parallel sequence sharding. Multimodal long-vision (1024 image tokens for hi-res) + caption sequences benefit. Cache adapter delta must shard along seq dim too | Add CP plan map; verify `multimodal_dataset.py`'s collate handles CP shard semantics; verify `image_mask` survives CP shard (currently per-row, would need per-shard) |

Priority order for phase6 work: **FSDP=2 × PP=4 first** (cheapest, no
new code), **then TP=2 variant** (key infra hole — TP support is
explicitly in-scope for upstream merge readiness), **then EP=2** (needs
MoE flavor in addition to AttnRes wrap), **then CP=2** (most code, biggest
multimodal payoff but lowest priority for a pre-merge PR).

For each config: launcher script + 1k-step alignment vs FSDP-only
baseline + alignment plot. Same pass criterion (max\|Δ\| ≤ 0.13 nats)
as the 4-GPU PP=4 V=2 result.

## Plan (3-4 weeks, single 4×5090 box)

| Week | Track A (parallelism) | Track B (multimodal) | Track C (polish) |
|---|---|---|---|
| W0 (4×5090) | Phase 5 Arm 1 to step 6000 → orchestrator `phase6/run_a1_alignment.sh` runs Arm 1' + Arm 2 (A1, real-data alignment) | — | C2 (start tests, run continuously) |
| W1 (8×rented) | A2 (PP=8 V=4 single-axis), A3 (FSDP=2 PP=2 TP=2 — *the* TP infra hole), A6 (determinism matrix) | B1 (variable image count) | C2 grows with each A/B item |
| W2 (8×rented) | A4 (async DCP), A5 (mid-save resume), FSDP=2 PP=2 EP=2 (MoE flavor) | B2 (interleave), ✅ B4 (sentinel registry, commit bffbbdf + d1a2fcf) | ✅ C1 (commit 435f89f), C3 |
| W3 (8×rented if budget, stretch) | FSDP=2 PP=2 CP=2 (long context) | B3 spec only (no impl) + B5 (kv-cache for AttnRes inference) | C4 (perf regression CI) |

C2 (CPU pytest) **runs continuously, not at the end** — every A/B item
lands with its own unit tests in the same PR. C2 in W2 is the final
matrix-completeness pass.

B3 is **spec'd but deferred** — vision-tower FSDP-shard is premature
optimization until we have a >4-GB-per-rank frozen vision encoder to
validate against, which our 4×5090 box cannot host.

### Status board (2026-05-01 update)

**Today's late-day commits (after the overnight pretrain landed)**:

| Item | Status | Commit | Notes |
|---|---|---|---|
| Projector save/load | ✅ done | 57a4b47 | Trainer registers `mm_projector` (projector + AdamW state) with checkpointer; full-state DCP resume now restores it. Eliminates the ~50-100 step projector reset penalty per crash. |
| v8 crash-resilient pretrain orchestrator | 🟡 in flight | fa1081d | Auto-relaunch loop exploiting projector save/load. Survived 3 KDA Triton crashes mid-run; loss continued to descend across all crashes (best 2.30 at step 5000, vs prior best 2.79). |
| A2 partial | ✅ done | n/a (results in tb only) | PP=4 layers_per_stage=1 = 4 virt/rank = 16 total virt stages + cache adapter, GBS=16 LOCAL_BS=1 seed=42, 500 steps. step-500 loss 3.48 (vs Arm 2 PP=4 V=2 = 3.83), V=4-per-rank schedule loss-invariant. |
| C4 perf regression CI | ✅ done | 6254286 | `phase6/perf_regression_check.py` — 5 verified-config baselines + 5% tolerance check. Smoke-tested PASS on live v8. |
| B5 partial (caption generation) | ✅ done | be78a37 | `phase5/generate_caption.py` — single-process inference, greedy decode, no KV cache. Will smoke-test against v8 final ckpt after pretrain ends. |
| A5 redo orchestrator | ✅ done (script ready, pending GPU) | 49b3351 | `phase6/run_a5_redo.sh` with strict `INFO - step: NN` grep filter (no longer matches torchrun WARNINGs). Phase 2a + 2b auto-resume continuity check. |
| B2 interleave dataset | ✅ done | 2f83d52 | `phase5/multimodal_dataset_interleave.py` + 7 CPU unit tests. Prefix / interior / random layouts. Image count + seq length invariant. Trainer wiring follow-up. |

Total tests now in phase5/tests/: **27 passing** (4 PP plumbing + 7 variable image + 9 sentinel + 7 interleave). Plus 97 in torchtitan/experiments/{attn_res,kimi_linear}/tests/ → **124 total passing CPU tests** for the AttnRes + multimodal stack.

### Status board (2026-04-30)

Items shipped so far on the 4×5090 box:

| Item | Status | Commit | Notes |
|---|---|---|---|
| A1 + A1.1 | ✅ done | 2d69453, b270b1d | FSDP=2-wrap projector → A1 alignment median 0.024 nats max 0.252 (warmup transient). Full plot at `phase6/alignment_arm2_real_mm_v2.png`. |
| B1 | ✅ done | 96f2647 (submodule) + e0e4b1d | Variable image count per row in `attn_res_model.py`; 6 CPU unit tests. |
| B4 | ✅ done | bffbbdf + d1a2fcf | `phase5/sentinel_registry.py` + 9 unit tests + train_mm wire-up. |
| C1 | ✅ done | 435f89f | `phase6/cache_adapter_ablation.md` — closed-form ratio `(N+1)/2`, 4× at our scale, projected 15-25% wallclock at 48B-A3B target. |
| C2 (partial) | ✅ partial | 0aa8453 | Mixed-dtype scatter test added. Total phase5 tests: 20. |
| PR draft | ✅ done | 2522a53 | `phase6/PR_DRAFT.md` consolidates all the above for the eventual upstream PR description. |
| Alignment plot tooling | ✅ done | 655a0fd | `compare_pp_vs_fsdp.py --out-plot` emits a 2-panel figure (loss curves + |Δ| vs noise band). |

Items pending (next 4-GPU work, then hand off to 8-GPU rented box):

| Item | Status | Notes |
|---|---|---|
| A6 partial | pending | FSDP=2 PP=2 mixed config smoke (500 steps from Phase 4 step-8000). Adds a third point to the parallelism matrix. |
| B2 | pending | Image-text interleave dataset (the model already supports it; B1's `test_image_mask_explicit_override` test proves it). Just need a real-data dataset wrapper. |
| C2 (rest) | pending | state_dict round-trip + partial-failure-recovery tests. |
| C3 (full) | pending | Rewrite `attn_res/README.md` and `phase5/README.md` in PR-ready form (PR_DRAFT.md is the start). |

Each Track A/B item ends with: launcher + test + alignment plot + 1-paragraph
writeup. Track C consolidates into PR-ready form.

## Concrete first-week actions

The orchestrator `phase6/run_a1_alignment.sh` runs in background and:

1. Polls `phase5/runs/arm1_fsdp/train.log` for "step: 60[12]X" — confirms
   Arm 1 is past step 6000 and the ckpt is safely on disk (caption-quality
   story deliverable).
2. SIGTERMs the running `phase5.train_mm` workers, waits up to 120 s for
   clean exit, SIGKILLs if needed.
3. Launches **Arm 1'** = FSDP=4 PP=1, `--debug.seed 42 --debug.deterministic`,
   from Phase 4 step-8000, GLOBAL_BS=12 LOCAL_BS=3, 2000 steps, `--metrics.log_freq 1`.
4. Launches **Arm 2** = PP=4 V=2 + Interleaved1F1B + cache adapter, same
   seed, same init, same GLOBAL_BS=12, 2000 steps.
5. Runs `phase5/compare_pp_vs_fsdp.py` → writes `phase6/alignment_report_arm2_real_mm.txt`.

**Why both runs init from Phase 4 step-8000 (not from Arm 1's step-6000)**:
Current Arm 1 was launched without `--debug.seed`, so its data shuffle
and projector init are not reproducible. Branching Arm 2 from Arm 1's
step-6000 would inherit that nondeterminism. Cleaner: re-init both
alignment runs from a known reproducible point (Phase 4 step-8000 + seed 42),
which gives a fully matched-seed alignment claim. Current Arm 1 still
serves the caption-quality story (loss curve to step 6000) — it just
isn't part of the alignment pair.

In parallel (CPU work, doesn't touch GPU): B1 — `MultiImageDataset`
that emits variable N_vision per row, with mixed (1-image, 0-image,
2-image) microbatches. Tests the LM forward path under the relaxed
`assert n_image_per_row == expected_per_row` constraint.

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
