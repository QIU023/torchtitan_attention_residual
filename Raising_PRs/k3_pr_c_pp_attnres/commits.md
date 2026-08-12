# PR-C backing work and extraction recipe

**Base**: PR-4025's head (`pr4025_latest`). See PR-A's `commits.md` for why the four
`k3_pr_*` branches are bookmarks rather than content.

## Blocked on an interface, not on effort

4025's decoder loop runs all layers from one entry point. PP needs it enterable at layer
`i` with `(x, block_residual_TND)` and exitable at `j` returning the same pair. Without
that seam this PR duplicates the loop body. The ask is filed on 4025. **Do not extract
before it is answered** -- the shape of the extraction depends on the answer.

Also relevant: the maintainer rejected upstreaming the cross-stage PP mechanism as a
GENERIC feature (~2026-04). This PR keeps it as private implementation inside the model
folder's parallelize path. Do not re-propose it as a generic mechanism.

## Extraction list

`pipeline_adapter.py` in full, 1578 lines. The parts a reviewer will ask about:

| symbol | lines | why it exists |
|---|---|---|
| `CrossStageCacheAdapter` | 447 | the adapter proper |
| `RankLocalCache` | 125 | same-rank reads across virtual stages under Interleaved1F1B |
| `pipeline_kimi_k3_with_cache_adapter` | 150 | entry point |
| `_install_mb_index_patch` | 71 | microbatch identity for the cache key |
| `_install_step_drop_patch` | 30 | step-end sweep; a FORWARD-ONLY microbatch never marks itself via backward |
| `_unwrap_multimodal_for_pp` | 87 | multimodal stage split |
| `_inject_kimi_k3_fqns` | 74 | checkpoint namespace shared with the wrapper |

DEP (`dep_enabled`, `dep_vision_stages`, `_install_vision_stage_wiring`,
`_dep_vision_share_index`, `_install_vision_prefetch`, `_register_mm_prefix_hooks`) is
~200 lines inside the same file. **Consider it a separate PR**: it is the report's sec
5.2.3 feature, it needs `n_vit > 1` to mean what the report says, and every measurement
here used `n_vit = 1`.

## Must NOT come along

* The three `components/` relaxations that let a frozen-only stage have no optimizer are
  CORE changes, not model changes. They are their own small PR and PR-C should depend on
  it rather than bundle it (`LORA_DEP_2026-08-11.md` for the four failure modes).

## Verification

Cells: `pp2`, `pp4`, `pp8`, `fsdp2_tp2_pp2`, `tp2_pp2_cp2`, `fsdp2_pp2_cp2`,
`ep2_fsdp2_tp2_pp2`, `ep2_fsdp2_pp2_cp2`, on BOTH the full-parameter and LoRA flavors --
18/18 each as of 2026-08-12. The exactness anchor is a shared checkpoint run at two PP
layouts; `PP_ADAPTER_AT_2P8T_2026-08-11.md` sec 2 documents the autograd hazard and why a
detached leaf is what holds.

## Status

Blocked on the 4025 decoder seam. Everything else is measured.
