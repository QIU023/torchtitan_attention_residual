# Backing commits — PR #2 base64 data-URL

## Discovered in

**Phase 11** — VLM rollout via SGLang Engine inside the GRPO chain.
Initial multimodal benchmark used file paths; switching to async rollouts
that handed inline image bytes from a torchstore-bridged actor mesh
revealed the missing data-URL path.

## Fork source

- Repo: `git@github.com:QIU023/sglang.git`
- Branch: `attention_residual_inference`
- Commit: `850ebb715`
- File touched: `python/sglang/srt/multimodal/processors/attn_res_vl.py` (one function, +6 / -0)

## Filing dependency

Blocks on **PR #5** (Block AttnRes inference overlay). `attn_res_vl.py`
is a fork-only file; it lands upstream as part of #5.

## Cherry-pick recipe (after PR #5 lands)

```bash
git checkout -b sglang-attn-res-vl-base64 upstream/main
git cherry-pick 850ebb715
git push origin sglang-attn-res-vl-base64
```

Alternative: fold this 6-line change into PR #5 as one of the
processor's initial features (cleaner for reviewers — one PR, one
working processor).
