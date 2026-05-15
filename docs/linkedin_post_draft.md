# LinkedIn post — draft

Three lengths below. All three link to (a) the RFC, (b) the fork. Pick whichever
fits the audience / how much detail you want to show.

---

## A. Short (≈ 80 words, fits the LinkedIn preview cleanly)

> I implemented Kimi Team's **Block Attention Residuals** (arXiv:2603.15031)
> as a reference implementation inside PyTorch's torchtitan, and filed RFC
> [pytorch/torchtitan#3029](https://github.com/pytorch/torchtitan/issues/3029).
>
> Reproduced the paper's loss-delta on a 174M Llama3 dense run, and built
> the cross-stage caching adapter that makes AttnRes pipeline-parallel under
> torchtitan's `Interleaved1F1B` schedule (paper §4.1) — validated on 4×
> RTX 5090.
>
> Upstream merge is gated on Kimi K3. Until then the fork is the canonical
> reference: https://github.com/QIU023/torchtitan/tree/attention_residual_dev
>
> (Detailed run logs and design notes are kept in a private logbook —
> available on request.)

---

## B. Medium (≈ 200 words, the version I'd actually post)

> **A reference implementation of Block Attention Residuals (Kimi Team, 2026)
> for PyTorch's torchtitan.**
>
> Block AttnRes (arXiv:2603.15031) replaces fixed residual addition with
> softmax attention over block outputs, using a per-layer zero-init pseudo-
> query. At N≈8 blocks it's pipeline-parallel-friendly: cross-stage traffic
> drops from O(L·d) to O(N·d).
>
> Over the last few weeks I ported the algorithm into torchtitan as an
> `experiments/` module, filed RFC #3029, and built two pieces of evidence:
>
> 1. **Single-GPU dense (174M Llama3, 20 k steps on C4):** AttnRes
>    consistently below baseline at every milestone — Δ −0.05 to −0.13,
>    consistent with the paper's "≈ baseline × 1.25 effective compute".
> 2. **PP cross-stage caching adapter (4-GPU PP=4 V=2):** naive-vs-adapter
>    loss delta stays inside the seed-vs-seed nondeterminism band over 1000
>    steps. The fix for the second-order backward through the cached tensor
>    was the non-obvious piece.
>
> Reviewers asked to gate upstream merge on Kimi's K3 release. Until then
> the fork is the reference impl — including a Kimi Linear (KDA + MLA + MoE)
> port for the K3-shape variant.
>
> RFC: https://github.com/pytorch/torchtitan/issues/3029
> Fork: https://github.com/QIU023/torchtitan/tree/attention_residual_dev
> Logbook: https://github.com/QIU023/AttnResidualTorchTitan
>
> #PyTorch #LLM #DistributedTraining #PipelineParallelism #OpenSource

---

## C. Long (≈ 350 words, if you want to use it as a portfolio writeup)

> **What I built: a reference implementation of Block Attention Residuals
> (Kimi Team, arXiv:2603.15031) for PyTorch's torchtitan training framework,
> and filed it as RFC pytorch/torchtitan#3029.**
>
> *Why AttnRes:* Standard residuals (`h_{l+1} = h_l + f_l(h_l)`) accumulate
> layer contributions with equal weight; depth dilutes shallow signal and
> grows hidden-state norm. AttnRes replaces the add with softmax attention
> over previous block outputs via a learned pseudo-query. At N≈8 blocks the
> cross-stage memory drops from O(L·d) to O(N·d) — making it the rare
> "architectural change that's actually pipeline-parallel friendly."
>
> *What's in the fork:*
>
> 1. **Algorithm.** `experiments/attn_res/` ships the primitive, zero-init
>    projection, and Llama3-shape + DeepSeek-V3-shape (MoE+MLA) flavors.
>    AttnRes lives as a torchtitan-native experiment, not a Llama3 subclass —
>    one-way `experiments → core` dependency only, no upstream changes.
> 2. **Evidence (single-GPU).** 174M Llama3 dense, 20 k steps on C4-en,
>    matched baseline vs AttnRes — Δ stays −0.05 to −0.13 across all
>    milestones, in line with the paper's loss-delta range.
> 3. **PP adapter.** Cross-stage caching adapter for `Interleaved1F1B`
>    (paper §4.1). The non-obvious bit was the second-order backward through
>    the cached tensor — fixed with `register_hook` + a detached-leaf cache.
>    Validated on 4× RTX 5090 PCIe at PP=4 V=2: |Δ_naive→adapter| ≤ 0.06
>    inside |Δ_naive→naive| ≤ 0.13 nondeterminism band over 1000 steps.
> 4. **Kimi Linear (Phase 4).** Full port of MoonshotAI/Kimi-Linear
>    (KDA + MLA + sigmoid-gated MoE) into torchtitan, with AttnRes wrapper
>    + PP adapter wired through. 436M FSDP overnight ran 12.5 k steps
>    successfully (architecture-grade, not pretraining-grade — that needs
>    H100-days).
>
> Reviewers on the RFC asked to gate upstream merge on the Kimi K3 release.
> The fork is the canonical reference until then — anyone who wants AttnRes
> in torchtitan can pull this rather than re-implement from the paper.
>
> Links:
> RFC: https://github.com/pytorch/torchtitan/issues/3029
> Fork: https://github.com/QIU023/torchtitan/tree/attention_residual_dev
> Logbook + writeups: https://github.com/QIU023/AttnResidualTorchTitan
>
> #PyTorch #LLM #DistributedTraining #PipelineParallelism #MoE #OpenSource

---

## Notes for posting

- **Image to attach** (any version): `phase2_attnres_baseline_loss/runs/comparison.png` — the
  baseline-vs-AttnRes loss curve. Most viewable proof-of-work in one image.
- LinkedIn truncates at ~210 chars before "see more"; version A is short
  enough that the full text is visible inline. Version B/C work better with
  the image attachment so the curve is the hook.
- All three intentionally avoid claiming "merged" or "shipped" — the
  framing is "reference implementation, upstream gated on K3".
