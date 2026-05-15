# Backing commits — PR #3 flashinfer_mla bf16 NaN

## Discovered in

**Phase 11** — first SGLang inference smoke for the Kimi-Linear AttnRes
1.4B-active ckpt (`phase11/eval_sft_3ep_qualitative.py`). The engine
booted, but the very first prefill returned NaN logits from the
deepest MLA layer. Switching to fp32 eager via the fork's
`attn_res_overlay` workaround unblocked the smoke immediately, but the
problem belongs upstream.

## Fork source

| Commit | Title | Files touched |
|---|---|---|
| `e8e7134ee` | `[AttnRes] fp32 MLA fallback: extend-only + write cache for native decode` | `python/sglang/srt/models/attn_res_overlay.py` |
| `334990612` | `[AttnRes] fp32 MLA eager fallback to fix flashinfer_mla NaN on Blackwell` | `python/sglang/srt/models/attn_res_overlay.py` |

Both commits on `QIU023/sglang@attention_residual_inference` (and
`main` after the merge to `dc154e785`).

## Status

- **Issue**: ready to file (see PR.md for issue body).
- **Patch**: depends on upstream API decision (kernel-level vs
  per-layer hook). Our `attn_res_overlay` implementation is the
  per-layer-hook reference; a kernel-level fix would obsolete it.

## Filing recipe

```bash
# 1. File the SGLang issue using PR.md body verbatim.
# 2. Cross-file a flashinfer issue with the same repro + the
#    kernel-level fix proposal.
# 3. WAIT for maintainer response on which API direction to land.
# 4. Once direction is chosen:
#    - If kernel-level: open a flashinfer PR for the kernel fix.
#    - If per-layer hook: cherry-pick e8e7134ee + 334990612 into a
#      hook-shaped PR on sglang, refactoring the AttnRes-specific
#      code into a general EagerFallbackRegistry API.
```

## Notes for the PR opener

- Make the issue cross-team-friendly: tag both `flashinfer-ai` and
  `sgl-project` maintainers.
- Include the exact RTX 5090 + SGLang commit + flashinfer version
  in the repro so maintainers don't have to chase a moving target.
- Don't submit our overlay's fork code as the patch verbatim — it's
  AttnRes-specific. Refactor into a general hook if the maintainers
  go that way.
