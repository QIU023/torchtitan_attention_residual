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

1. The maintainer's constraint: **AttnRes cannot be bolted onto an
   existing pretrained LLM**. The pseudo-query (paper §5) is
   zero-init and slowly learned during pretraining; retrofitting it
   onto a frozen, already-converged model breaks its optimization
   premise — at best a no-op, at worst a perturbation.

2. The hardware constraint: 4× RTX 5090 PCIe, no path to a real 7B+
   AttnRes pretraining run.

3. The narrative constraint: the project's value is *AttnRes
   architecture in a real training framework*, not "best multimodal
   benchmark." We need an experiment where AttnRes carries its own
   weight as a from-scratch training story.

## What the maintainer's rule allows

The rule is "don't retrofit onto a pretrained LLM." It does NOT
forbid AttnRes inside any *new* component that you train from
scratch within a multimodal stack — even when the LLM backbone is a
frozen public model.

Three valid placements:

```
┌──────────────────────────────────────────────────────────────────┐
│              Path A — AttnRes in the Connector                    │
│                                                                   │
│   SigLIP / DINOv2 (frozen)                                        │
│           │                                                       │
│           ▼                                                       │
│   ┌─────────────────────┐                                         │
│   │ AttnRes Connector   │  N transformer blocks + AttnRes,        │
│   │  (train from scratch)│  trained on caption data               │
│   └─────────────────────┘                                         │
│           │                                                       │
│           ▼                                                       │
│   [aligned text-space tokens]                                     │
│           │                                                       │
│           ▼                                                       │
│   Llama-3.1-8B / Qwen3-7B (frozen or LoRA)                        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│              Path B — AttnRes in cross-attention adapter          │
│                                                                   │
│   Llama-3.1-8B (frozen base)                                      │
│         │                                                         │
│         │ + AttnResAdapter blocks inserted between LLM layers,    │
│         │   trained from scratch, attend to vision tokens         │
│         ▼                                                         │
│   multimodal-aware logits                                         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│              Path C — AttnRes in a from-scratch ViT               │
│                                                                   │
│   image → ViT(AttnRes between blocks) → SigLIP-style embeds       │
│   (only viable with massive image data; out of scope for          │
│    4× 5090 hardware)                                              │
└──────────────────────────────────────────────────────────────────┘
```

## Recommended: Path A — AttnRes Connector

This is the cleanest fit for the project's hardware + narrative
constraints.

### Why Path A wins for this project

* **Connector is small** (~0.5–1B params, 4–8 transformer blocks at
  hidden 1024-2048). 4× 5090 trains it overnight on caption data.
* **Connector is from-scratch** → AttnRes's optimization story
  applies in full.
* **Strong LLM backbone is available** without us having to train it
  (frozen Llama-3.1-8B, Qwen3-7B, etc.).
* **A clean A/B exists**: connector-with-AttnRes vs vanilla-connector,
  both trained on the same caption data with the same frozen vision
  + LLM. This is the *publishable* claim — "AttnRes improves
  multimodal connector training," same shape as the original
  AttnRes paper's dense LM claim.
* **Maintainer's rule satisfied**: backbone is untouched.
* **PP adapter story stays cleanly separate**: the connector is
  small enough to not need PP, so it doesn't entangle with the
  Phase 3/4 PP work.

### Concrete shape (typical LLaVA-style connector dimensions)

```
input:    (B, n_image_tokens=576, vision_dim=1152)   # SigLIP-SO400M
          │
          │ Linear projection → (B, 576, hidden=2048)
          ▼
   ┌──────────────────────────────────────┐
   │ Block 1: self-attn + MLP + AttnRes_1 │
   ├──────────────────────────────────────┤
   │ Block 2: self-attn + MLP + AttnRes_2 │
   ├──────────────────────────────────────┤
   │       ...                             │
   ├──────────────────────────────────────┤
   │ Block N: self-attn + MLP + AttnRes_N │
   └──────────────────────────────────────┘
          │
          │ Linear → (B, 576, llm_hidden=4096)
          ▼
output:   text-space embedded image tokens, ready to splice into
          LLM input sequence (LLaVA convention: image tokens
          precede text tokens)
```

`AttnRes_i` is the same per-paper recipe used in
`torchtitan/experiments/attn_res/`: a pseudo-query w_l of shape
(D, 1), zero-initialized, attended over cumulative block outputs.

### Training recipe

Mirrors LLaVA-1.5 stage 1 (connector-only pretrain):

* **Frozen**: vision tower (SigLIP-SO400M), LLM (Llama-3.1-8B-Base)
* **Trainable**: connector blocks + AttnRes pseudo-queries
* **Data**: 558K LLaVA caption pairs (CC3M + LAION subset, public)
* **Loss**: standard LM loss over caption tokens, image tokens
  positioned via LLaVA's `vision_token_id=-200` sentinel (already
  scaffolded in `torchtitan/experiments/kimi_linear/multimodal_model.py`)
* **Hardware**: 4× 5090 single-node, FSDP across 4 ranks for the
  connector + LLM (LLM frozen so weights replicate cheaply, only
  cache management).
* **Steps**: ~5K-10K, ~6h overnight on 4× 5090.

### A/B experimental plan

Two arms:

| arm | connector | expected outcome |
|---|---|---|
| `vanilla_connector` | N=8 transformer blocks, no AttnRes | baseline LLaVA caption loss |
| `attn_res_connector` | N=8 transformer blocks + AttnRes after each | lower caption loss + better downstream VQA, mirroring AttnRes paper's LM claim |

Compare:
1. Caption loss curves over training
2. Zero-shot VQA accuracy on a held-out set (e.g. VQAv2 dev)
3. CLIP-style retrieval accuracy on COCO captions

If the AttnRes connector measurably wins on (1) + (2), that's the
publishable contribution. If it doesn't, that's also a clean
negative result ("AttnRes's LM gains do not transfer to small
caption connectors") — still useful for the writeup.

## Where the existing 436M Kimi Linear ckpt fits in this story

**Not as the main multimodal LM backbone.** Use Llama-3.1-8B-Base
(frozen) for that.

But the 436M ckpt is still useful as a **secondary "small-LM head"
in a plumbing experiment**, *not* a benchmark target:

* Verify the connector's outputs compose correctly with a small
  trainable LM (drives end-to-end gradient flow through the full
  multimodal stack including a from-scratch student LM).
* Test the cache adapter on the multimodal sequence (vision + text
  tokens) under PP=4, Problem B's adapter is wired and the adapter
  is content-agnostic.
* Demonstrate the integration between Phase 3/4 PP+adapter work and
  the multimodal scaffolding — same fork, same submodule, same
  trainer. That cohesion is the core "Kimi Linear + AttnRes +
  multimodal full stack" claim from `multi_modal_idea.md`.

The 436M is the **plumbing demo backbone**. The Llama-3.1-8B is
the **performance demo backbone**. Different roles, no conflict.

## What the deliverables look like

End of project:

1. **PP adapter system validation** — already complete (Phase 3 +
   Phase 4 + Problem B). Three-arm comparison, loss alignment, eval.
2. **AttnRes architecture port to Kimi Linear** — already complete
   (Phase 4c–4e).
3. **Multimodal scaffolding** — already wired in
   `torchtitan/experiments/kimi_linear/multimodal_model.py` (LLaVA-
   style projector, vision_token_id=-200 sentinel).
4. **AttnRes Connector module** — to be added under
   `torchtitan/experiments/attn_res/connector/` (new). Architecture
   mirrors LLaVA's projector + adds AttnRes between blocks.
5. **A/B caption-pretrain run** — vanilla vs AttnRes connector,
   ~6h overnight each on 4× 5090 with frozen SigLIP + frozen
   Llama-3.1-8B-Base.
6. **Multimodal eval** — VQAv2 zero-shot or COCO retrieval, ~30
   min on 4× 5090.

KD/MiniPLM negative results documented for honesty (see
`docs/pretraining_closure_and_kd_plan.md`); they are not on the
critical path.

## Decision summary (for the project closure RFC)

> AttnRes's optimization premise (zero-init pseudo-query trained
> alongside the rest of the model) cannot be retrofitted onto a
> pretrained LLM. For multimodal applications it must therefore be
> placed in a component that is itself trained from scratch within
> the multimodal stack. The cleanest such component is the
> vision↔LLM connector. We propose an `AttnResConnector` module
> trained on standard LLaVA caption data with frozen vision (SigLIP)
> and frozen LLM (Llama-3.1-8B-Base) backbones, with a clean A/B
> against a vanilla connector to test whether the AttnRes
> architectural advantage transfers to the multimodal alignment
> regime. This satisfies the maintainer's "no retrofit" rule,
> respects 4× RTX 5090 hardware bounds, and produces a publishable
> A/B in ~12h of overnight training (two arms × 6h each).
