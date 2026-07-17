# Kimi K3 release — impact on this repo (2026-07-16)

> **Event.** MoonshotAI announced **Kimi K3** on 2026-07-16 (blog + API live;
> full open weights scheduled **2026-07-27**; formal tech report not yet out).
> K3's blog **officially confirms Attention Residuals (AttnRes) as a production
> architecture component** — the exact thesis this repo bet on since Phase 2.
> This doc records what K3 confirms, what it does *not* yet reveal, and the
> pre-registered checklist to reconcile our implementation the moment the
> weights/config/report drop.

## 0. Why this doc exists

Once K3's config.json + tech report + (likely) reference inference code are
public, there will be an **official AttnRes implementation** and probably
third-party reproductions. This repo's standing claim is *"first open-source
implementation of Block Attention Residuals inside torchtitan."* That
first-mover position has a shelf life measured from **2026-07-27**. This doc
maximizes our defensibility with the information available **today**.

## 1. What K3 officially confirms (verified, multi-source)

Cross-checked across Moonshot's blog, Moonshot's X account, MarkTechPost,
VentureBeat, the-decoder, and latent.space (see Sources). Consistent facts:

| Claim | K3 official statement | Our repo's prior basis |
|---|---|---|
| **AttnRes is real + production** | *"built on two architectural updates: Kimi Delta Attention (KDA) and Attention Residuals (AttnRes)"* | We implemented from arXiv:2603.15031 (preprint) — now vendor-confirmed |
| **AttnRes gain** | *"boost training efficiency by about 25 percent while adding less than 2 percent in extra compute overhead"* | We cited *"paper-claimed ~1.25x effective-compute"* — now upgradable to **officially confirmed ~25% / <2% overhead** |
| **AttnRes mechanism** | *"selectively retrieves representations across depth rather than accumulating them uniformly"* | Exactly our `block_attn_res` softmax-over-prior-blocks vs. uniform residual add |
| **KDA** | hybrid linear attention, *"up to 6.3× faster decoding in million-token contexts"*; refinement of Gated DeltaNet | We ported KDA via `fla-core` in `experiments/kimi_linear/` |
| **Scaling** | *"~2.5× better scaling than K2"* (KDA + AttnRes + sparsity + recipes) | — |

**The headline: our project's central number (1.25×) is now corroborated by the
model vendor, not just a preprint.** The `<2% extra compute overhead` figure is
*new* and worth adopting.

## 2. K3 scale vs. our carriers — NOT the same model

K3 is a **2.8T-parameter, A50B, 896-expert (16 activated)** frontier model. Our
carriers are deliberately small (hardware-bound):

| | K3 (2026) | Our `kimi_linear_48b` flavor | Our trained carrier |
|---|---|---|---|
| Total params | 2.8T | 49.1B | 447M |
| Experts | 896 (16 act) | 256 (8 act) | grouped-topk |
| Attention | KDA + **Gated MLA** | KDA + MLA | KDA + MLA |
| AttnRes | yes (N undisclosed) | Block N≈8 | Block N=4 |

**Implication:** K3 is a *bigger sibling* of the Kimi-Linear paper architecture
(arXiv:2510.26692) we ported, not a different family. Our 48B flavor tracks the
**Kimi-Linear-48B-A3B** reference config, which predates K3 — it is still a
valid Kimi-Linear-shape carrier. We do **not** claim to reproduce K3; we claim
to be an independent open implementation of the AttnRes + KDA mechanisms K3
productionizes.

## 3. K3 architecture extras BEYOND what we ported (the gap surface)

K3 stacks several components that are **not part of AttnRes** and **not in the
Kimi-Linear paper** we ported. These are the honest "we don't have this yet"
list — none invalidate our AttnRes work, but expect them in the report:

| K3 component | What it is | Our status |
|---|---|---|
| **Gated MLA** | gated variant of Multi-head Latent Attention | We have plain MLA (DSv3-faithful NoPE) — **gap** |
| **Quantile Balancing** | expert allocation from router-score quantiles, no aux-loss heuristic | We use sigmoid grouped-topk — **gap** |
| **Per-Head Muon** | Muon optimizer extended to per-attention-head adaptation | We use AdamW — **gap** |
| **SiTU (Sigmoid Tanh Unit)** | activation for control | We use SwiGLU — **gap** |
| **Stable LatentMoE** | K3's MoE framework | We use torchtitan stock MoE — **gap** |
| **MXFP4 weights / MXFP8 activations** | QAT from SFT stage | We did fp8 rowwise experiments — **partial** |

**None of these are AttnRes.** They are orthogonal K3 innovations. Our
contribution is scoped to **AttnRes + the PP cross-stage cache adapter + KDA/MLA
port**, and that scope stays intact.

**Graftability onto a *pretrained* Kimi-Linear-48B base** (for the [phase 13
plan](phase13_k3like_48b_posttrain/PLAN.md)): **AttnRes** (zero-init, safe),
**Quantile Balancing** (routing, training-time), and **Per-Head Muon** (optimizer)
graft cleanly. **Gated MLA** is hard (changes attention structure), **SiTU** is
not graftable (weights trained for SwiGLU), **Stable LatentMoE** is unknown until
the report. So the realistic "K3-like" model = **Kimi-Linear-48B + AttnRes**
(+ optional Quantile Balancing / Per-Head Muon).

**Distributed-infra completeness (audited 2026-07-16 @ torchtitan `90d85eba3`):**
model-side 5D is code-complete — `parallelize_kimi_linear` implements FSDP2/HSDP,
TP (DSv3-style, KDA=NoParallel, fla-core DTensor patch), EP (all-to-all), and the
PP `pipelining_fn` adapter; only **CP** is missing and it is upstream-blocked
(fla-core `chunk_kda` lacks ring-recurrence). Gaps are not primitives but
*composition* (full-5D-at-once untested off multi-H200) and *veRL integration
glue* (does veRL's torchtitan worker drive `pipelining_fn`; train<->rollout
weight reshard; rollout-side AttnRes serving). At 48B, FSDP+TP+EP holds the model
without PP, so veRL post-training runs without the PP-integration gap.

## 4. Pre-registered reconciliation checklist (execute when weights drop 2026-07-27)

The moment K3's `config.json` / `modeling_*.py` / tech report is public, diff
against our implementation in this exact order. Pre-registering it here proves
our implementation predates K3's public code.

- [ ] **AttnRes block count N.** K3 blog withholds N. Compare K3's actual N (and
      per-block layer grouping) to our `num_blocks` sweep (N∈{2,3,4,6,8,12}).
      Our bet: N≈8 sweet spot. → `experiments/attn_res/attn_res.py:AttnResConfig`
- [ ] **AttnRes variant.** Full (N=L) vs Block (N<L) — which did K3 ship? We
      implement both; Block is our headline. → `attn_res_model.py`
- [ ] **Pseudo-query design.** K3 blog: query "decoupled from hidden state."
      Confirm our zero-init `AttnResProjection` (D→1) matches K3's learned query
      shape/init. → `attn_res.py:AttnResProjection`
- [ ] **AttnRes placement.** Per-layer (2× per transformer block: pre-attn +
      pre-FFN) vs per-block-boundary. Confirm against K3's forward. Our port
      does 2× per block. → `kimi_linear/attn_res_model.py:KimiAttnResDecoderLayer`
- [ ] **The `<2% overhead` measurement.** Reproduce K3's overhead definition:
      it is the *algorithm's* extra FLOPs (naive-AttnRes vs no-AttnRes baseline),
      **NOT** our PP-adapter's communication bookkeeping (+2.7% step-time at
      48B-shape on PCIe — a different quantity; see §5).
- [ ] **KDA gating.** K3 = "finer-grained gating" refinement of Gated DeltaNet.
      Diff our `fla-core` KDA gate against K3's. → `kimi_linear/model.py:KimiDeltaAttention`
- [ ] **Gated MLA.** Diff K3's gated MLA against our plain MLA; scope a follow-up
      if the gate is load-bearing for AttnRes interaction.
- [ ] **Inference two-phase.** If K3 ships reference inference, diff their block
      aggregation against our SGLang two-phase overlay (online-softmax exact
      merge). Our `<11% prefill TTFT reduction` was measured on 447M; compare.

## 5. Numbers to keep straight (so we're not caught flat-footed)

K3's "**<2% compute overhead**" and our pressure-test "**+2.7% step-time**" are
**different quantities** — do not let a reviewer or competitor conflate them:

- **K3's <2%** = extra *arithmetic* from the AttnRes aggregation itself
  (softmax over prior blocks), measured as AttnRes-model vs. no-AttnRes baseline
  at matched params. This is the *algorithm* overhead.
- **Our +2.7% step-time** (48B-shape PP=8×VP=4) = the *PP cross-stage cache
  adapter's* communication + bookkeeping cost on a **PCIe** fabric, measured as
  adapter vs. naive PP (both with AttnRes). At the same shape the adapter also
  gave **−11.4% peak memory**. This is a *pipeline-integration* overhead, and on
  PCIe the bookkeeping exceeds the bandwidth saving (the saving converts to
  wall-clock only on NVLink / inter-node).

Both are honest and both are small; they just answer different questions. Our
repo can now *additionally* report the algorithm-overhead number in K3's sense
(naive-AttnRes vs baseline) if we want a direct apples-to-apples with K3 — we
have the baseline flavors registered to measure it.

## 6. Actions triggered by this release

**Documentation currency (do now — the gate these docs waited on is met):**
- The torchtitan AttnRes RFC (#3029) and sglang PR #5 were explicitly *gated on
  "Kimi K-series release."* **K3 is that release.** The legitimacy anchor now
  exists (cite the K3 blog). Recommend: file the PR #5 RFC **after 2026-07-27**
  weights drop for the strongest anchor (announced-but-not-open today).
- Upgrade "paper-claimed ~1.25x" → "confirmed in Kimi K3 (2026): ~25% training
  efficiency, <2% compute overhead" across repo docs + resume, with the K3 blog
  as citation.

**Competitive positioning (do now):**
- Timestamp the first-mover claim: our fork + this repo predate K3's public
  code (weights 2026-07-27). This doc + git history are the timestamp.
- Keep the scope honest: we implement **AttnRes + PP adapter + KDA/MLA port**,
  not K3's Gated-MLA / Quantile-Balancing / Per-Head-Muon / SiTU / LatentMoE.

**Deferred to 2026-07-27 (weights/report drop):**
- Execute the §4 reconciliation checklist.
- Decide whether Gated MLA / the true N warrant a code follow-up.
- If K3 ships reference inference, diff against our SGLang two-phase overlay.

**Deferred to "rent GPUs" (user):**
- Re-run pretraining aligned to whatever recipe the report reveals.

## 7. One-line summary

**K3 vendor-confirms AttnRes (~25% gain / <2% overhead) and KDA — validating this
repo's entire thesis. No code change is forced today; the real reconciliation is
the §4 checklist executed when weights drop 2026-07-27. Until then: upgrade the
1.25×→confirmed language, un-gate the AttnRes RFC/PR #5, and hold the honest
scope line (we did AttnRes + PP adapter + KDA/MLA, not K3's other five
innovations).**

## Sources

- [Kimi K3 Tech Blog — Open Frontier Intelligence](https://www.kimi.com/blog/kimi-k3)
- [Moonshot AI Releases Kimi K3 (MarkTechPost, 2026-07-16)](https://www.marktechpost.com/2026/07/16/moonshot-ai-releases-kimi-k3-a-2-8-trillion-parameter-open-moe-model-with-kimi-delta-attention-and-1m-context/)
- [China's Moonshot AI releases Kimi K3 (VentureBeat)](https://venturebeat.com/technology/chinas-moonshot-ai-releases-kimi-k3-the-largest-open-source-model-ever-rivaling-top-u-s-systems)
- [Kimi's open model K3 nears GPT-5.6 (the-decoder)](https://the-decoder.com/kimis-open-model-k3-nears-gpt-5-6-sol-and-fable-5-while-signaling-the-end-of-super-cheap-chinese-ai/)
- [Kimi K3 2.8T-A50B (latent.space AINews)](https://www.latent.space/p/ainews-kimi-k3-28t-a50b-the-largest)
- [Kimi-Linear paper (arXiv:2510.26692)](https://arxiv.org/pdf/2510.26692) — the architecture family we ported
