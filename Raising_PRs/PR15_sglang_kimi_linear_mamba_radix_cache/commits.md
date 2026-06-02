# Backing commits — PR #15 Kimi-Linear → MambaRadixCache

## Discovered in

**Phase 11 / overnight GRPO (2026-06-01..02).** GRPO rollouts on our 447M
Kimi-AttnRes VLM came out fluent but **image-blind** (free-association text
unrelated to the image), so the GQA reward collapsed to ~3% while the same
checkpoint scored ~45% in offline eval. A 6-step elimination (capacity ceiling →
advantage-collapse → prompt format → image format → batched-async → **radix**)
isolated the cause to the **radix cache**: with `disable_radix_cache=True` the
model was grounded; with radix ON it went blind.

Code root-cause: KDA's recurrent **SSM state** is incompatible with plain
`RadixCache` prefix reuse (`kda_backend.py` `has_initial_state = extend_prefix_lens
> 0` reads an uncheckpointed prefix-boundary state). The fork had wired KDA's
backend + `HybridLinearKVPool` but **not** the prefix-cache selection
(`is_hybrid_ssm` in `scheduler.py` never matched Kimi-Linear), so it silently used
plain `RadixCache`. Mainline additionally **force-disables** radix for
`KimiLinearForCausalLM` in `server_args.py`.

## Fork source

| Field | Value |
|---|---|
| Repo | `git@github.com:QIU023/sglang.git` |
| Branch | `attention_residual_inference` |
| Backing commit | `c3b588019` — *"feat(attn_res): MoE score-before parity (bug B) + Kimi-Linear->MambaRadixCache (Fix A)"* |
| Upstream base | `3da87902d` (sglang PR #23013) — the state validated against |

## What to cherry-pick / hand-port

`c3b588019` bundles **two** unrelated changes. For PR #15 take ONLY the radix
parts; the MoE score-before parts are a separate concern (our checkpoint's
training convention, **not** for upstream):

**Include (PR #15):**
- `python/sglang/srt/configs/kimi_linear.py` — the appended
  `register_linear_attn_model(LinearAttnModelSpec(... uses_mamba_radix_cache=True,
  unwrap_text_config=True ...))` block (near the end of the file).
- `python/sglang/srt/server_args.py` — removing `KimiLinearForCausalLM` from the
  force-`support_mamba_cache=False` elif (~L2149).

**Exclude (NOT PR #15):**
- `python/sglang/srt/configs/kimi_linear.py` — the `moe_score_before_experts`
  config field (belongs to the bug-B score-before line; not upstream-relevant).
- `python/sglang/srt/models/attn_res_overlay.py` — entirely (score-before MoE).

Because both the register block and the `moe_score_before_experts` field live in
`kimi_linear.py`, hand-port the register block onto a clean `upstream/main`
checkout rather than cherry-picking the whole file:

```bash
git checkout -b pr15-kimi-mamba-radix upstream/main
# 1) add register_linear_attn_model(...) for KimiLinearConfig in configs/kimi_linear.py
# 2) drop KimiLinearForCausalLM from the force-disable elif in server_args.py
# 3) confirm: type(scheduler.tree_cache).__name__ == "MambaRadixCache" with radix ON
```

## Validation artifacts (this fork)

- Official `moonshotai/Kimi-Linear-48B-A3B-Instruct` (bf16, TP2): MambaRadixCache
  selected, registry double-hit, `ssm_state 32.81GB` allocated, radix-ON vs
  `--disable-radix-cache` greedy **4/4 byte-identical** (incl. shared-long-prefix).
- 447M Kimi-AttnRes VLM: radix-ON blind→grounded (GQA ~3%→~45%) after the wiring;
  smoke A/B 4/4 identical. (Repro scripts were under `/home/seqkd_overnight/`.)

## Status

- 🟢 Backing commit pushed to `QIU023/sglang @ attention_residual_inference`.
- Propose-and-hand-port (register block + elif removal) onto `upstream/main`.
- Re-verify mainline still lacks it before filing (likely open per #12867 inactive).
