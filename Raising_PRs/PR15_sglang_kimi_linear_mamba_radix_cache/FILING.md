# PR #15 — filing instructions

## Status

🟢 **Branch pushed; validated on official `moonshotai/Kimi-Linear-48B-A3B-Instruct`; PR not yet opened.**

| Item | Link / value |
|---|---|
| Fork branch | https://github.com/QIU023/sglang/tree/pr15-kimi-mamba-radix |
| Open-PR URL | https://github.com/QIU023/sglang/pull/new/pr15-kimi-mamba-radix |
| Target repo | https://github.com/sgl-project/sglang |
| Base | `sgl-project/sglang:main` |
| Head | `QIU023/sglang:pr15-kimi-mamba-radix` |
| Commit | `1f532d4b3` (1 commit, +52/-5 across `configs/kimi_linear.py` + `server_args.py`) |
| Base | branched directly off `upstream/main` HEAD `ed85bcf8c` (2026-05-21); no rebase needed; py_compile clean |
| Validation | **4/4 byte-identical greedy A/B** on official 48B incl. shared-long-prefix pair; MambaRadixCache selected at scheduler |
| Cross-link | #12867 (target), #11214 (MambaRadixCache v0), #22326 (checkpointing), #20415 (unified hybrid radix refactor) |

## To open the PR

1. Open https://github.com/QIU023/sglang/pull/new/pr15-kimi-mamba-radix
2. Confirm base = `sgl-project/sglang:main`, head = `QIU023/sglang:pr15-kimi-mamba-radix`
3. Title + body below (the body pulls heavily from [PR.md](PR.md) which has the full validation tables)
4. Cross-link to #12867 in the PR body and as a "Closes" line if appropriate
5. Submit

---

## Title (copy-paste)

```
[scheduler/configs] Wire Kimi-Linear (KDA) into MambaRadixCache and stop force-disabling its radix cache
```

## Body (copy-paste — abridged from PR.md; full validation tables there)

```markdown
## Summary

`KimiLinearForCausalLM` today runs in sglang only with the radix cache
**force-disabled**. With radix enabled it produces fluent but
context-detached output: a prefix-cache hit makes the KDA backend read
the recurrent **SSM state** from a slot that was never checkpointed at
that prefix boundary, so the reused prefix's state is garbage.

The fix is **not** a new mechanism. sglang already merged `MambaRadixCache`
(#11214) which checkpoints/forks recurrent state at radix nodes — Qwen3-Next /
GDN already use it. Kimi-Linear was simply never registered into that path,
and is in fact actively force-disabled. This PR registers it
(`register_linear_attn_model(...)` block in `configs/kimi_linear.py`) so
`is_hybrid_ssm=True` → `MambaRadixCache`, and removes the force-disable
branch in `server_args.py`. Listed as a wanted-but-undone target in #12867.

## Root cause

1. `layers/attention/linear/kda_backend.py` — `has_initial_state = extend_prefix_lens > 0`.
   On a radix prefix hit, KDA reads the prefix-boundary recurrent state from the
   `ssm_states` / `conv_states` slot and only computes the suffix.
2. `managers/scheduler.py` — `is_hybrid_ssm` only enables `MambaRadixCache`
   for `register_linear_attn_model(uses_mamba_radix_cache=True)` registrants
   (plus the legacy `hybrid_gdn_config` / `mamba2_config` paths). Kimi-Linear
   matches none, so it falls back to plain `RadixCache`, which never
   checkpoints the SSM state → the slot KDA reads is stale.
3. `server_args.py` puts `KimiLinearForCausalLM` in the elif that forces
   `support_mamba_cache=False` (radix off), suppressing even the fallback.

## The fix

**(1)** `configs/kimi_linear.py` — append a `register_linear_attn_model(
LinearAttnModelSpec(...))` block at the bottom of the module (it imports
at registry-discovery time, before `ServerArgs._handle_model_specific_adjustments`
runs). Spec: `config_class=KimiK3Config`, `backend_class_name` = lazy
str to `KDAAttnBackend`, `arch_names=["KimiLinearForCausalLM"]`,
`uses_mamba_radix_cache=True`, `support_mamba_cache=True`,
`support_mamba_cache_extra_buffer=False` (MambaRadixCache v1 has no
extra-buffer path for KDA yet — keep the no_buffer page_size=1 +
overlap-off route), `unwrap_text_config=True`.

**(2)** `server_args.py` — drop `KimiLinearForCausalLM` from the elif
that calls `_handle_mamba_radix_cache(support_mamba_cache=False)`. The
registry path at the top of the function now correctly calls
`_handle_mamba_radix_cache(support_mamba_cache=True)` → `page_size=1` +
overlap-off + `MambaRadixCache`. Re-running the legacy elif would
disable radix again. Comment inline explains the change.

No KDA backend / kernel / mem_cache changes are needed. State snapshot
in the `no_buffer` path is produced by `cache_unfinished_req` /
`cache_finished_req` forking the request's own pool slot; prefix match
restores it with `mamba_pool.copy_from` — both backend-agnostic.
GDN-style `_track_mamba_state_extend` is a no-op when
`enable_mamba_extra_buffer()` is False (the path this PR selects).

Verified caveats (no patch needed):
- `page_size==1` — Mamba-radix v0 default/auto already selects this.
- Overlap-schedule auto-disabled for `no_buffer`.
- `mamba_cache_chunk_size = max(FLA_CHUNK_SIZE=64, page_size) = 64`
  matches KDA kernel `chunk_size=64`.
- `mamba2_layer_cache(layer_id)` keyed by
  `KimiK3Config.mamba2_cache_params.layers == linear_layer_ids`
  (same global LM index KDA passes).

## Validation — on the official `moonshotai/Kimi-Linear-48B-A3B-Instruct`

Downloaded bf16 (98.25 GB, 20 shards), booted in patched sglang, TP=2
(45.84 GB/GPU), radix **enabled**.

### MambaRadixCache selected at scheduler (both ranks)

```
[FIXA_PROBE] tree_cache=MambaRadixCache  is_hybrid_ssm=True  disable_radix_cache=False
             linear_attn_spec=LinearAttnModelSpec:uses_mamba_radix_cache=True  page_size=1
```
Registry double-hit:
```
get_linear_attn_spec_by_arch('KimiLinearForCausalLM') -> ('KimiK3Config', uses_mamba=True)
get_linear_attn_config(<48B cfg>)                     -> ('KimiK3Config', uses_mamba=True)
```
Startup log:
```
Mamba Cache is allocated. ... ssm_state size: 32.81GB
```

### Radix-ON correctness equals radix-OFF (greedy, 4/4 byte-identical)

4 text prompts including one shared-long-prefix pair (an Apollo-program
paragraph followed by two distinct questions → guaranteed prefix-cache
hit).

| prompt | radix ON (`MambaRadixCache`, this PR) | radix OFF (`ChunkCache`, `--disable-radix-cache`) | match |
|---|---|---|---|
| Apollo … "Who first walked on the Moon?" | `Neil Armstrong` | `Neil Armstrong` | ✅ |
| Apollo … "What year?" | `1969` | `1969` | ✅ |
| (prompt 3) | *(identical)* | *(identical)* | ✅ |
| (prompt 4) | *(identical)* | *(identical)* | ✅ |

**4/4 byte-identical greedy output** including the shared-prefix pair →
the SSM state checkpoint / restore in `MambaRadixCache` is numerically
exact for KDA; prefix reuse does not corrupt the recurrent state.

## Test plan

- [x] `py_compile` on both patched files (Python 3.12).
- [x] Boot `KimiLinearForCausalLM` with radix enabled, assert
      `type(scheduler.tree_cache).__name__ == "MambaRadixCache"` and
      `is_hybrid_ssm` is True. *(verified — see scheduler probe above)*
- [x] Shared-long-prefix greedy A/B vs `--disable-radix-cache` →
      byte-identical (4/4). *(verified — table above)*
- [ ] Regression: non-Kimi model's cache selection unchanged (registry
      is arch-gated; default `False`). *(static-reviewed; no behavior
      change for any arch not in `arch_names`)*

## Backwards compatibility

The patch only changes radix-cache *selection* for
`KimiLinearForCausalLM`. No kernel / backend / mem_cache changes.
Existing `--disable-radix-cache` continues to work. Other models'
cache selection is unchanged (registry is arch-gated).

## Related upstream issues / PRs

- #12867 — Hybrid Linear LLMs support (lists Kimi-Linear as wanted)
- #11214 — MambaRadixCache v0 (the merged mechanism this PR wires Kimi-Linear into)
- #22326 — Mamba radix checkpointing
- #20415 — Unified hybrid radix refactor
```

## Cross-links to other PRs in this batch

- **PR #1** (`SGLANG_DISABLE_SHM_MM`) — separate sglang PR, same fork
- **PR #7** (KDA `causal_conv1d` fp16 type-join) — separate sglang PR, same fork
- **PR #8** (fp8 MoE Blackwell shmem) — separate sglang PR, same fork
- **PR #5** (Block AttnRes overlay) — research-track; gated on Kimi K-series release; this PR is independent of it (only touches the upstream `KimiLinearForCausalLM` path)

## Reviewer hints

- The registration block in `configs/kimi_linear.py` lives at module
  import time and exercises `linear_attn_model_registry`'s public
  `register_linear_attn_model()`. No private API touched.
- The `server_args.py` change removes a 5-line elif and adds a 9-line
  explanatory comment. The functional delta is "stop calling
  `_handle_mamba_radix_cache(support_mamba_cache=False)` for
  `KimiLinearForCausalLM`."
- Validation was deliberately performed on the canonical released model
  (`moonshotai/Kimi-Linear-48B-A3B-Instruct`), not a custom fork
  checkpoint, to prove this is not a private-ckpt hack.
