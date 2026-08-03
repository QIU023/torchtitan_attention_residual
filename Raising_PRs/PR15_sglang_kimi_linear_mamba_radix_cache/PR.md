# PR #15 — Register Kimi-Linear (KDA) for `MambaRadixCache`; stop force-disabling its radix cache

> (= entry **#13** in `UPSTREAM_PR_LIST.md`; folder numbered 15 to avoid the existing `PR13*` fla-kernel folders.)

**Target repo**: `sgl-project/sglang`
**Target paths**: `python/sglang/srt/configs/kimi_linear.py`, `python/sglang/srt/server_args.py`
**Fork reference**: `QIU023/sglang @ attention_residual_inference`, HEAD `c3b588019`
**Effort**: ~0.5–1.5 day (2-line-class change + validation; MambaRadixCache infra already exists)
**Risk**: low — only changes radix-cache *selection* for `KimiLinearForCausalLM`; reuses the existing, merged `MambaRadixCache`. No kernel/backend changes.

---

## Suggested PR title

> [scheduler/configs] Wire Kimi-Linear (KDA) into `MambaRadixCache` and stop force-disabling its radix cache

---

## Summary

`Kimi-Linear` (KDA linear-attention + MLA) runs in sglang today **only with the
radix cache force-disabled**. With radix enabled it produces fluent but
context-detached / corrupted output: a prefix-cache hit makes the KDA backend
read the recurrent **SSM state** from a slot that was never checkpointed at that
prefix boundary, so the reused prefix's state is garbage.

The fix is **not** a new mechanism. sglang already merged `MambaRadixCache`
(#11214) which checkpoints/forks recurrent state at radix nodes — Qwen3-Next/GDN
already use it. Kimi-Linear simply was **never registered into that path**, and is
in fact **actively force-disabled**. This PR registers it (two-line-level change)
so `is_hybrid_ssm=True` → `MambaRadixCache`, and removes the force-disable branch.
Listed as a wanted-but-undone target in #12867.

## Root cause (code-grounded)

1. `layers/attention/linear/kda_backend.py` — `has_initial_state = extend_prefix_lens > 0`.
   On a radix prefix hit, KDA reads the prefix-boundary recurrent state from the
   `ssm_states`/`conv_states` slot and only computes the suffix.
2. `managers/scheduler.py` (`is_hybrid_ssm` selection) only enables
   `MambaRadixCache` for `hybrid_gdn_config` / `mamba2_config` /
   `register_linear_attn_model(uses_mamba_radix_cache=True)`. **Kimi-Linear matches
   none**, so it falls back to plain `RadixCache`, which never checkpoints the SSM
   state → the slot KDA reads is stale/cross-request.
3. `server_args.py` puts `KimiLinearForCausalLM` in the elif that forces
   `support_mamba_cache=False` (→ radix off), so even the fallback is suppressed.

Net: the fork wired KDA's **backend + `HybridLinearKVPool`**, but the **prefix
cache selection** was never wired → backend on the mamba route, prefix cache on
the plain route = mismatch.

## The fix (detailed)

**(1) `configs/kimi_linear.py`** — register Kimi-Linear into the linear-attn
registry at import time (the module is imported transitively before
`ServerArgs._handle_model_specific_adjustments` runs):

```python
register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=KimiK3Config,
        backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",  # lazy str, no import cycle
        arch_names=["KimiLinearForCausalLM", "KimiBlockAttnResForCausalLM",
                    "KimiAttnResVLForConditionalGeneration"],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=False,   # no_buffer path
        unwrap_text_config=True,                  # VLM carrier configs resolve inner KimiK3Config
    )
)
```

**(2) `server_args.py`** — drop `KimiLinearForCausalLM` from the elif that forces
`_handle_mamba_radix_cache(support_mamba_cache=False)` (the registry path at the
top of the function now handles it correctly).

**No KDA backend / kernel / mem_cache changes are needed.** In the `no_buffer`
path the state snapshot is produced by `cache_unfinished_req` / `cache_finished_req`
forking the request's own pool slot, and prefix match restores it with
`mamba_pool.copy_from` — both backend-agnostic. (GDN-style `_track_mamba_state_extend`
is a no-op when `enable_mamba_extra_buffer()` is False.)

Verified caveats (no patch needed): `page_size==1` (Mamba-radix v0; default/auto
already 1), overlap-schedule auto-disabled for no_buffer, `mamba_cache_chunk_size =
max(FLA_CHUNK_SIZE=64, page_size)=64` matches the KDA kernel `chunk_size=64`, and
`mamba2_layer_cache(layer_id)` is keyed by `KimiK3Config.mamba2_cache_params.layers
== linear_layer_ids` (same global LM index KDA passes).

---

## Validation — on the **official** `moonshotai/Kimi-Linear-48B-A3B-Instruct`

To prove this is not a private-checkpoint hack, the fix was validated on the
canonical released model (text MoE, 27 layers = 20 KDA + 7 MLA, 256-expert),
**not** our custom AttnRes VLM. Downloaded bf16 (98.25 GB, 20 shards), booted in
the patched sglang, TP=2 (45.84 GB/GPU), radix **enabled**.

### Example data 1 — MambaRadixCache is now selected (scheduler probe, both ranks)

```
[FIXA_PROBE] tree_cache=MambaRadixCache  is_hybrid_ssm=True  disable_radix_cache=False
             linear_attn_spec=LinearAttnModelSpec:uses_mamba_radix_cache=True  page_size=1
```
Registry double-hit (loaded via the engine's own `get_config`):
```
get_linear_attn_spec_by_arch('KimiLinearForCausalLM') -> ('KimiK3Config', uses_mamba=True)
get_linear_attn_config(<48B cfg>)                     -> ('KimiK3Config', uses_mamba=True)
```
Startup log corroboration (plain RadixCache never allocates this):
```
Mamba Cache is allocated. ... ssm_state size: 32.81GB
```

### Example data 2 — radix-ON output is correct AND identical to radix-OFF

4 text prompts incl. one **shared-long-prefix** pair (an Apollo-program paragraph
followed by two different questions → guaranteed prefix-cache hit). Greedy decode.

| prompt | radix ON (`MambaRadixCache`) | radix OFF (`ChunkCache`, `--disable-radix-cache`) | match |
|---|---|---|---|
| Apollo … "Who first walked on the Moon?" | `Neil Armstrong` | `Neil Armstrong` | ✅ |
| Apollo … "What year?" | `1969` | `1969` | ✅ |
| (prompt 3) | *(identical)* | *(identical)* | ✅ |
| (prompt 4) | *(identical)* | *(identical)* | ✅ |

**4/4 byte-identical greedy output**, including the shared-prefix pair → the SSM
state checkpoint/restore in `MambaRadixCache` is numerically exact for KDA; prefix
reuse does not corrupt the recurrent state.

### Example data 3 — the symptom this fixes (our 447M Kimi-AttnRes VLM)

Same model+prompt+image, radix ON, **before** the fix (plain RadixCache) vs after
(`MambaRadixCache`):

| | rollout output for an image VQA prompt | GQA |
|---|---|---|
| radix ON, plain RadixCache (broken) | *"a guide to the best restaurants in town… In 1946 the Soviet Union…"* (image-blind, context-detached) | ~3% |
| radix ON, MambaRadixCache (this PR) | short grounded answers ("Color", "Large", "No") | ~45–48% |
| radix OFF (ChunkCache) | grounded; A/B 4/4 identical to MambaRadixCache | ~45% |

### Conclusion of validation

- **Mainline lacks this** and actively force-disables Kimi-Linear radix (see
  `commits.md`); this PR is the only path for official Kimi-Linear to use
  `MambaRadixCache` prefix reuse. Benefits the canonical 48B, not just our model.
- Correctness is exact (4/4 identical greedy vs the no-cache path).

---

## Test plan (for the PR)

1. `KimiLinearForCausalLM` boots with radix enabled and selects `MambaRadixCache`
   (assert `type(scheduler.tree_cache).__name__ == "MambaRadixCache"`, `is_hybrid_ssm`).
2. Shared-long-prefix greedy A/B: radix-ON output == `--disable-radix-cache` output.
3. Same-prompt `n>1` sampling stays coherent (no prefix-hit corruption).
4. Regression: a non-Kimi model's cache selection is unchanged (registry is
   arch-gated; default `False`).

## Filing checklist

- [ ] Re-check **latest sglang HEAD** — confirm Kimi-Linear is still unregistered /
      still force-disabled (verdict here is vs fork base `3da87902d` = PR #23013;
      #12867 being inactive suggests it's still open).
- [ ] Cross-link #12867 (target), #11214 (MambaRadixCache v0), #22326 (checkpointing),
      #20415 (unified hybrid radix refactor).
- [ ] Port the 2 changes onto a clean `upstream/main` branch (see `commits.md`).
- [ ] Add the shared-prefix A/B test above to the PR's test section.

## References
- sglang #12867 (Hybrid Linear LLMs support — lists Kimi-Linear), #11214, #22326, #20415, PR #23013
- PyTorch blog: *Hybrid Models Meet SGLang — More Than Full Attention*
- Model: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
