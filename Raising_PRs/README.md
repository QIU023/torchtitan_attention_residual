# Upstream PR drafts — ready to file

Per-PR filing kit for upstream contributions surfaced during this project.
Each subfolder contains:

- **`PR.md`** — full PR description (title, summary, motivation, patch, test plan, filing checklist).
- **`commits.md`** — discovery phase + backing fork commits + cherry-pick recipe (or "propose-only" note when no fork commit exists).

Filing flow per PR:

1. Read `commits.md` for the backing commit hash + cherry-pick recipe.
2. Prepare an upstream fork branch (`git checkout -b <pr-branch> upstream/main`).
3. Cherry-pick / hand-port / isolate the backing commits per `commits.md`.
4. Push and open the PR using `PR.md` body verbatim.

---

## Current PRs (11 candidates)

| # | Folder | Target | Title | Discovered | Status |
|---|---|---|---|---|---|
| 1 | `PR1_sglang_disable_shm_mm/` | sglang | `SGLANG_DISABLE_SHM_MM` env to force CPU mm transport | Phase 11 | ✅ Ready — clean cherry-pick (`74083ffae`) |
| 2 | `PR2_sglang_base64_data_url/` | sglang | base64 data-URL in `attn_res_vl` image loader | Phase 11 | 🟡 Ready (`850ebb715`); blocks on PR #5 |
| 3 | `PR3_sglang_flashinfer_mla_bf16_nan/` | sglang + flashinfer | flashinfer_mla bf16 NaN repro + fp32 fallback knob | Phase 11 | 🟠 Issue ready; patch shape pending upstream API decision |
| 4 | `PR4_torchtitan_parallelize_fn_signature/` | torchtitan | `parallelize_fn` signature stability for `experiments.rl.PolicyTrainer` | Phase 11 | 🟡 Propose-only (no fork commit; hand-port from `PR.md`) |
| 5 | `PR5_sglang_attn_res_inference_overlay/` | sglang | Block AttnRes inference overlay (Kimi + Qwen3 carriers) | Algorithm root Phase 2-4; inference path Phase 11 | 🟠 Research-track; needs Kimi K-series release for legitimacy |
| 6 | `PR6_sglang_rs_merge_ag_seq_shard_fusion/` | sglang | RS+merge+AG seq-shard fusion as a documented feature | Phase 11 | 🟠 RFC; depends on PR #5 + ≥1 other adopter |
| 7 | `PR7_sglang_kda_causal_conv1d_fp16/` | sglang | KDA `causal_conv1d_triton` fp16 dtype type-join fix | Phase 11 | ✅ Ready; bundled in `a6c46168a` — isolation recipe in `commits.md` |
| 8 | `PR8_sglang_fp8_moe_blackwell_shmem/` | sglang | fp8 weight-only MoE fused kernel Blackwell shmem autotune | Phase 11 | 🟡 Partial (launch path) bundled in `a6c46168a`; downstream ICA needs followup |
| 9 | `PR9_sglang_attn_res_einsum_cublas_bypass/` | sglang + cuBLAS | AttnRes block-aggregation einsum → broadcast+sum (cuBLAS bypass) | Phase 11 | 🟡 Ready bundled in `a6c46168a`; blocks on PR #5; cuBLAS root cause is a separate driver issue |
| 10 | `PR10_sglang_fp8moe_ignored_layers_warning/` | sglang | `Fp8Config.get_quant_method` user-visible warning on silent bf16 fallback | Phase 11 | 🔵 Tentative — file only if PR #8 downstream ICA stays unresolved long-term |
| 11 | `PR11_torchstore_sync_endpoint_dispatch/` | pytorch/torchstore | sync-endpoint dispatch policy: async caller opt-in | Phase 11 | 🟠 Issue ready; patch shape pending upstream API decision |

**Legend**: ✅ ready to file · 🟡 ready but conditional (depends on another PR or hand-port) · 🟠 issue / RFC first, patch deferred · 🔵 tentative / wait

PR #12 (engine-agnostic `Generator` abstraction + SGLang reference impl)
is tracked separately in [`../UPSTREAM_PR_LIST.md`](../UPSTREAM_PR_LIST.md)
as RFC-track work; not yet in `Raising_PRs/` since it depends on PR #4
landing first.

---

## Suggested filing order

After the 2026-05-15 inventory refresh:

1. **PR #1** — smallest possible first contribution; clean cherry-pick. Builds reviewer credibility.
2. **PR #4** — 1-day torchtitan PR; required prerequisite for the future Generator PR (UPSTREAM_PR_LIST.md PR #12).
3. **PR #7** — parallel with #4; independent kernel fix.
4. **PR #3 issue** (no patch) — let flashinfer team weigh in on API direction.
5. **PR #9 issue** for cuBLAS reproducer + the overlay-side patch (after PR #5 lands or as part of #5).
6. **PR #11 issue** for torchstore (API design discussion).
7. **PR #8** with partial shmem-shrink + ICA-acknowledgement.
8. **PR #6** as a design RFC (no code change yet).
9. **PR #5** RFC after Kimi K-series release; staged landing (algorithm → carrier → VL).
10. **PR #2** folded into PR #5, or as a 6-line follow-up.
11. **PR #10** if/when PR #8's downstream ICA is resolved (otherwise skip).

---

## Phase discovery summary

| Phase | What it surfaced |
|---|---|
| **Phase 2** | Original Block AttnRes algorithm work (174M dense Llama3 paper-Table-1 reproduction). Roots of PR #5. |
| **Phase 4** | Kimi Linear MoE port + AttnRes wrapper. Roots of PR #5's carrier shape. |
| **Phase 11** | Inference + RLHF push. Surfaced **everything else** — PR #1, #2, #3, #4, #6, #7, #8, #9, #10, #11, and the inference-path side of PR #5. |

The cluster on Phase 11 reflects how integration work (production inference + RL training) is where upstream rough edges surface — the algorithm-stage Phases 2-4 ran inside torchtitan's training path which is more mature.

---

## What goes here vs. `UPSTREAM_PR_LIST.md`

- **`../UPSTREAM_PR_LIST.md`** — long-form inventory + design discussion across all PR candidates. Lives at repo root.
- **`Raising_PRs/<PR>/`** — actionable filing kit for the PR being prepared. Once a PR is filed, add the upstream PR number to that PR's folder; once merged, the folder can be archived or deleted.
