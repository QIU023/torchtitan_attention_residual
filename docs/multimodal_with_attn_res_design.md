# Architecture: applying Attention Residual to multimodal — without
# retrofitting onto pretrained LLMs

Companion to `multi_modal_idea.md` and `pretraining_closure_and_kd_plan.md`.
That earlier set assumed the 436M Kimi Linear student would itself act
as the multimodal LM backbone after distillation. After running the
KD experiments (online KD, then MiniPLM-style data distillation —
both ending at c4 val_loss ~3.82, slightly *worse* than pre-KD's
3.73), it is now clear that the 436M ckpt is **at the floor** for the
4× 5090 + c4 + Llama-tokenizer training budget. It is excellent for
closing the "Kimi Linear architecture port + PP adapter system
validation" project, but is not a viable LM for downstream multimodal
benchmarks.

This doc lays out the *coherent* way to keep doing AttnRes work in a
multimodal setting, given:

1. **Maintainer's actual rule** (sharper restatement after iteration):
   *AttnRes must be co-pretrained from scratch with the model it
   wraps.* The pseudo-query (paper §5) is zero-init and slowly
   learned during pretraining; bolting it into a Llama3 / Qwen3 /
   etc. model definition that has NOT been pretrained from scratch
   with AttnRes is exactly what's forbidden — the optimization
   premise is broken. **The rule does NOT prevent us from using
   AttnRes inside any new component we train from scratch as part
   of the multimodal stack.**

2. The hardware constraint: 4× RTX 5090 PCIe, no path to a real 7B+
   AttnRes pretraining run.

3. The narrative constraint: the project's value is *AttnRes
   architecture in a real training framework*, not "best multimodal
   benchmark." We need an experiment where AttnRes carries its own
   weight as a from-scratch training story — and that component
   needs to be **substantial** enough for the AttnRes optimization
   advantage to show up (a tiny 4-block connector is too thin).

## Earlier paths (kept as appendix; superseded)

The first iteration of this doc proposed three placements:

* **Path A — AttnRes Connector**: a small (4-8 block, 0.5-1B params)
  transformer between frozen SigLIP and frozen Llama-3.1-8B. Easy
  hardware fit but **the connector is too small for AttnRes to
  meaningfully demo** — the optimization advantage is most
  pronounced in deep transformer training, marginal in shallow
  alignment modules.
* **Path B — AttnRes cross-attention adapter**: LoRA-style adapter
  blocks inside frozen Llama. **Risky workaround**: even though the
  adapter blocks themselves are from-scratch, they're inserted into
  Llama's pretrained layer stack — getting close to "retrofitting
  into the LLM definition" the maintainer flagged.
* **Path C — AttnRes ViT from scratch**: most paper-aligned but
  needs LAION-400M+ image data and weeks on H100 — out of scope
  for 4× 5090.

After feedback, the better-targeted options below replace these.

## Sharper options

Three categories of placement that actually pass the
"substantial + co-pretrained from scratch + multimodal-relevant"
bar.

### Category I — AttnRes inside a substantial from-scratch
### multimodal sub-model

#### Path D — Q-Former / Perceiver Resampler (1.5-3B, deep, recommended)

The Q-Former (BLIP-2) and Perceiver Resampler (Flamingo) sit between
a frozen vision tower and a frozen LLM, but unlike a connector they
are **deep cross-attention transformers** (24-32 blocks at hidden
2560-4096), often the dominant trainable component of the entire
multimodal model. They are always trained from scratch.

```
SigLIP-SO400M (frozen)
     │
     ▼
Q-Former / Perceiver Resampler   ← TRAIN FROM SCRATCH
     ┌──────────────────────────────────────────┐
     │ N learnable query tokens                 │
     ├──────────────────────────────────────────┤
     │ Block 1: SelfAttn(Q) +                    │
     │          CrossAttn(Q ↔ vision_tokens) +   │
     │          MLP + AttnRes_1                  │
     ├──────────────────────────────────────────┤
     │ Block 2: ... + AttnRes_2                  │
     ├──────────────────────────────────────────┤
     │ ... 24-32 blocks total                    │
     ├──────────────────────────────────────────┤
     │ Block 32: ... + AttnRes_32                │
     ├──────────────────────────────────────────┤
     │ Linear: hidden → llm_dim                  │
     └──────────────────────────────────────────┘
     │
     ▼
[N text-space "soft tokens"]
     │ injected at <image> position
     ▼
Llama-3.1-8B-Base (frozen)

Trainable params: 1.5-3B  (deep enough for AttnRes to demo)
Hardware: FSDP across 4× 5090, overnight feasible at small B/seq.
```

**Why D > A:**
* Deep transformer (24-32 layers vs 4-8) → AttnRes's optimization
  advantage actually shows up.
* Q-Former is the **performance-determining component** of BLIP-2
  / InstructBLIP — it's not glue, it's the model.
* Co-pretrained from scratch on caption + VQA data ⇒ AttnRes
  pseudo-queries learn alongside the rest, paper-aligned.
* Standard, replicable A/B against vanilla Q-Former.

#### Path E — From-scratch Visual Decoder Head

If multimodal *generation* (text → image, à la CM3leon / Chameleon)
is on the table, the visual decoder head is a from-scratch
trained transformer that converts LLM hidden states back into image
tokens. Substantial (~500M-1B params, deep). AttnRes between blocks.

Same constraint applies as Path C: needs significant image-token
training data. We do not pursue this without a proper image-tokenized
corpus.

### Category II — Use the existing 436M AttnRes ckpt as a
### specialist module in a multimodal system

These reuse the AttnRes-from-scratch investment we already paid for
in Phase 4. The 436M ckpt was co-pretrained with AttnRes from
scratch on c4; that property is preserved when we use the ckpt as a
component. No retrofit, no maintainer violation.

#### Path F — Speculative-decoding draft model (recommended pair with D)

Speculative decoding for multimodal inference: a frozen "main"
multimodal model (e.g. LLaVA-Next-8B) verifies K-token candidate
sequences proposed by a fast "draft" model. Draft quality directly
determines throughput.

```
Inference:
  Big multimodal model (frozen, expensive forward)   ← LLaVA-Next-8B
     ▲                                                 or similar
     │  K candidate tokens (verify forward = 1 pass)
  ┌──┴──────────────────────────────────────────┐
  │ Draft model = our 436M AttnRes-Kimi (Phase 4)│  ← AttnRes
  │   accepts visual tokens too                  │     trained
  └──────────────────────────────────────────────┘     from scratch
     │
     ▼
  Tokens accepted up to first rejection, resample
  rejected position, continue.
```

**Why F is compelling:**
* Reuses 436M ckpt **exactly as it was trained** (no retrofit).
* AttnRes-Kimi vs vanilla draft (Llama-3.2-1B etc.) is a direct
  A/B on a metric that *matters in production*: speculative
  decoding acceptance rate / wall-clock speedup.
* Multimodal twist: extend the draft to accept image tokens
  (small modification of multimodal_model.py vision_token_id=-200
  scaffolding).
* AttnRes's claimed "better trained representations" → "better
  draft predictions" → "higher acceptance rate." Clean, measurable.
* No need to retrain or distill the 436M; it's used as-is.

#### Path G — Multimodal reward model / verifier

In RLHF for multimodal, a reward model (or output verifier) is
trained from scratch on preference data. The 436M Kimi
(or a fresh from-scratch AttnRes model) is a fine candidate
backbone for the reward head. AttnRes participates in that
from-scratch reward-model pretraining.

#### Path H — MoE router / dispatcher

In a mixture-of-experts multimodal serving setup, a small router
chooses between specialists. The router is from-scratch trained on
routing decisions. 436M Kimi as router → AttnRes contribution
preserved. Lower-impact than F or G; lower priority.

### Category III — Train a complete small multimodal model from scratch

#### Path J — Mini-LLaVA (~1B end-to-end with AttnRes everywhere)

The most paper-aligned but most expensive path: train a small
multimodal model where vision encoder, connector, AND text decoder
are all from-scratch, all with AttnRes:

```
vision encoder (300M, AttnRes-ViT, from scratch)
       ▼
connector (200M, AttnRes)
       ▼
text decoder (500M, AttnRes-Kimi-style, from scratch)
```

Trained on 5-10B tokens of caption + interleaved data. Needs 5-10
overnights on 4× 5090. Real but expensive. Would not beat
LLaVA-1.5 in performance, but is a *clean* end-to-end AttnRes
multimodal demonstration.

## Comparison table

| path | feasibility on 4×5090 | reuses 436M ckpt | AttnRes substantive? | engineering cost | recommend |
|---|---|---|---|---|---|
| **D. Q-Former / Resampler** | ✅ overnight | ❌ new 1-3B from scratch | **strong** (deep) | medium-high | **★★★** |
| **F. Speculative draft** | ✅ overnight | **✅ as-is** | **strong** (uses already-trained AttnRes ckpt) | medium | **★★★** |
| G. Reward model | ✅ overnight | ✅ | medium | medium | ★★ |
| J. Mini-LLaVA full | ❌ 5-10 overnights | partial | strongest | high | ★ |
| E. Visual decoder | ⚠️ data-bound | ❌ | strong | high | ★ |
| H. MoE router | ✅ but weak demo | ✅ | weak | medium | ★ |
| ~~A. tiny connector~~ | ✅ | ❌ | weak (too small) | low | (superseded) |
| ~~B. cross-attn adapter~~ | ⚠️ | ❌ | medium | high | (close to retrofit) |
| ~~C. ViT from scratch~~ | ❌ | ❌ | strongest | high | (out of scope) |

## Recommended: Path D + Path F in parallel

The two complementary stories that close the project most
coherently:

**Path F — speculative draft** showcases the *practical value of
the 436M AttnRes investment we already made.* AttnRes-trained
Kimi 436M predicts tokens better than a vanilla 1B draft → higher
acceptance rate in speculative decoding → wall-clock speedup on
multimodal inference. Direct, measurable, reuses the existing
ckpt without any retraining.

**Path D — Q-Former / Perceiver Resampler** showcases that *AttnRes
remains useful when training a substantial new multimodal component
from scratch.* A/B against vanilla Q-Former on the same caption
corpus. This is the closest analog to the AttnRes paper's
dense-LM claim, but in cross-modal alignment.

Both paths share infrastructure:
* Same frozen vision tower (SigLIP-SO400M)
* Same frozen large LLM (Llama-3.1-8B-Base or LLaVA-Next-8B)
* Same multimodal scaffolding (`multimodal_model.py` already
  scaffolds `vision_token_id=-200` LLaVA convention)
* Same caption dataset (LLaVA's 558K pretrain pairs)
* Same evaluation harness (VQAv2 zero-shot, COCO retrieval, plus
  speculative-decoding benchmarks for path F)

The 436M Kimi ckpt has TWO concrete roles in this design, both
respecting the maintainer's rule:

1. **Path F**: deployed as-is as the speculative draft model.
2. **Path D side-experiment**: serves as a small "verify-the-pipeline"
   LM head behind the trained Q-Former, since LLaVA-Next-8B may be
   too heavy for some smoke runs. Plumbing demo, not benchmark.

## Out-of-scope items (kept for future H100-class work)

* Train a 7B+ AttnRes LM from scratch
* Train a from-scratch ViT with AttnRes
* Mini-LLaVA full end-to-end
* AttnRes inside Llama's frozen layer stack (this would violate the
  maintainer's rule)

## Project-closure deliverable list (updated)

1. **PP adapter system validation** — complete (Phase 3 + Phase 4 +
   Problem B).
2. **AttnRes architecture port to Kimi Linear** — complete (Phase 4c-4e).
3. **KD/MiniPLM negative result writeup** — complete (Phase 5).
4. **Path F: speculative-decoding draft A/B** — Phase 6 (this dir).
   * AttnRes-Kimi-436M draft vs Llama-3.2-1B draft.
   * Frozen LLaVA-Next-8B (or Llama-3.1-8B-Base for text-only ablation).
   * Acceptance rate + wall-clock speedup on COCO captioning + VQAv2.
5. **Path D: Q-Former / Perceiver Resampler A/B** — Phase 6.
   * AttnRes Q-Former vs vanilla Q-Former.
   * Frozen SigLIP + frozen LLM, train on LLaVA-Pretrain.
   * Caption loss curves + VQAv2 zero-shot accuracy.

## Decision summary (project closure RFC excerpt)

> AttnRes's optimization premise — a zero-init pseudo-query that
> co-trains with the surrounding model from scratch — forbids
> bolting AttnRes into a pretrained LLM's layer definition (the
> maintainer's rule). It does not forbid AttnRes inside any
> from-scratch trained component of a multimodal system, nor does
> it forbid using a model that *was* AttnRes-co-pretrained as a
> downstream specialist module.
>
> Two paths satisfy these constraints, fit 4× RTX 5090 hardware,
> and produce publishable A/B comparisons:
>
> * **Path F (speculative-decoding draft model)**: deploy our
>   already-trained AttnRes-Kimi-436M as the draft for a frozen
>   large multimodal model. A/B against a same-tokenizer vanilla
>   1B draft. Metric: acceptance rate, wall-clock speedup.
>
> * **Path D (Q-Former / Perceiver Resampler)**: train a
>   substantial (1.5-3B) deep cross-attention transformer with
>   AttnRes between blocks, on top of frozen SigLIP + frozen
>   Llama-3.1-8B-Base. A/B against vanilla Q-Former. Metric:
>   caption loss, downstream VQAv2 accuracy.
>
> Together they cover both directions AttnRes might add value:
> reusing already-trained AttnRes weights (F), and producing a new
> deeply-trained AttnRes module within the multimodal stack (D).
> Each path requires ~1-2 overnights on the existing hardware.
