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

## Current PRs (11 active + 1 obsoleted)

| # | Folder | Target | Title | Discovered | Status |
|---|---|---|---|---|---|
| 1 | `PR1_sglang_disable_shm_mm/` | sglang | `SGLANG_DISABLE_SHM_MM` env to force CPU mm transport | Phase 11 | 🚀 **Branch pushed** → [pr1-disable-shm-mm](https://github.com/QIU023/sglang/tree/pr1-disable-shm-mm). PR not yet opened. See [`PR1.../FILING.md`](PR1_sglang_disable_shm_mm/FILING.md). |
| 2 | `PR2_sglang_base64_data_url/` | sglang | base64 data-URL in `attn_res_vl` image loader | Phase 11 | 🟡 Ready (`850ebb715`); blocks on PR #5 |
| 3 | `PR3_sglang_flashinfer_mla_bf16_nan/` | sglang + flashinfer | flashinfer_mla bf16 NaN repro + fp32 fallback knob | Phase 11 | 🟠 Issue ready; FILING.md drafted with target URLs (sglang + flashinfer cross-link). |
| 4 | `PR4_torchtitan_parallelize_fn_signature/` | torchtitan | `parallelize_fn` signature stability for `experiments.rl.PolicyTrainer` | Phase 11 | ⛔ **OBSOLETED-BY-UPSTREAM** (2026-05-17). `627f4a31 [rl] Trainer refactor` already landed the widening on 2026-04-20. Fork rebase tracked in [`FORK_REBASE_TASK.md`](FORK_REBASE_TASK.md). |
| 5 | `PR5_sglang_attn_res_inference_overlay/` | sglang | Block AttnRes inference overlay (Kimi + Qwen3 carriers) | Algorithm root Phase 2-4; inference path Phase 11 | 🟠 Research-track; needs Kimi K-series release for legitimacy |
| 6 | `PR6_sglang_rs_merge_ag_seq_shard_fusion/` | sglang | RS+merge+AG seq-shard fusion as a documented feature | Phase 11 | 🟠 RFC; depends on PR #5 + ≥1 other adopter |
| 7 | `PR7_sglang_kda_causal_conv1d_fp16/` | sglang | KDA `causal_conv1d_triton` fp16 dtype type-join fix | Phase 11 | 🚀 **Branch pushed** → [pr7-kda-causal-conv1d-fp16](https://github.com/QIU023/sglang/tree/pr7-kda-causal-conv1d-fp16). GPU smoke verified on RTX 4070Ti SM 8.9 (6/7 cases). PR not yet opened. See [`PR7.../FILING.md`](PR7_sglang_kda_causal_conv1d_fp16/FILING.md). |
| 8 | `PR8_sglang_fp8_moe_blackwell_shmem/` | sglang | fp8 weight-only MoE fused kernel Blackwell shmem autotune | Phase 11 | 🚀 **Branch pushed** → [pr8-fp8-moe-blackwell-shmem](https://github.com/QIU023/sglang/tree/pr8-fp8-moe-blackwell-shmem). Static-verified (`py_compile` OK; GPU smoke needs SM 12.0). PR not yet opened. See [`PR8.../FILING.md`](PR8_sglang_fp8_moe_blackwell_shmem/FILING.md). |
| 9 | `PR9_sglang_attn_res_einsum_cublas_bypass/` | pytorch (cuBLAS) + sglang | AttnRes block-aggregation einsum → broadcast+sum (cuBLAS bypass) | Phase 11 | 🟠 **Re-scoped 2026-05-17**: file cuBLAS root-cause issue on `pytorch/pytorch` NOW; sglang overlay patch blocked on PR #5. See [`PR9.../FILING.md`](PR9_sglang_attn_res_einsum_cublas_bypass/FILING.md). |
| 10 | `PR10_sglang_fp8moe_ignored_layers_warning/` | sglang | `Fp8Config.get_quant_method` user-visible warning on silent bf16 fallback | Phase 11 | 🔵 Tentative — file only if PR #8 downstream ICA stays unresolved long-term |
| 11 | `PR11_torchstore_sync_endpoint_dispatch/` | pytorch/torchstore | sync-endpoint dispatch policy: async caller opt-in | Phase 11 | 🟠 Issue ready; FILING.md drafted. |
| 14 | `PR14_torchtitan_opd_trainer_on_policy_distillation/` | torchtitan | `OPDTrainer` (GKD on-policy distillation) sibling to `PolicyTrainer` | Phase 11 | 🟡 Design + commits.md drafted; foundation (TRL JSD adapter + HF teacher scorer) validated on fork; trainer assembly (C.2/C.3) pending |

**Legend**: 🚀 branch pushed · 🟢 branched locally (push pending) · ✅ ready to file (no local branch yet) · 🟡 ready but conditional (depends on another PR or hand-port) · 🟠 issue / RFC first, patch deferred · 🔵 tentative / wait · ⛔ obsoleted

**Filing convention**: each folder has `PR.md` (PR body draft) + `commits.md` (backing commit hashes / cherry-pick recipe) + `FILING.md` (target URLs, copy-paste title/body, cross-links between PRs in this batch). PR-body content lives in `PR.md`; `FILING.md` is the actionable execution doc.

**Internal (non-PR) maintenance**: [`FORK_REBASE_TASK.md`](FORK_REBASE_TASK.md) — fork rebase required after upstream torchtitan `627f4a31` refactor; blocks GRPO launchers from running on freshly-pulled fork until reconciled.

PR #12 (engine-agnostic `Generator` abstraction + SGLang reference impl)
is tracked separately in [`../UPSTREAM_PR_LIST.md`](../UPSTREAM_PR_LIST.md)
as RFC-track work. **Unblocked as of 2026-05-17** — the upstream
`627f4a31` refactor that obsoleted PR #4 satisfies #12's prerequisite
(non-Qwen3 model_specs can now drive `PolicyTrainer._build_model`
without per-launcher adapter shims). Not yet in `Raising_PRs/` because
it's RFC-track (needs upstream design discussion on the Generator
interface) rather than direct file-and-cherry-pick.

---

## Suggested filing order

Refreshed 2026-05-17 after PR #4 obsoleted-by-upstream:

1. **PR #1** — smallest possible first contribution; clean cherry-pick. Already branched locally; just push + open PR. Builds reviewer credibility.
2. **PR #7** — independent Triton kernel fix; already branched locally (file-isolated from bundle `a6c46168a`); push + open PR after GPU smoke.
3. **PR #3 issue** (no patch) — let flashinfer team weigh in on API direction.
4. **PR #11 issue** for torchstore (API design discussion).
5. **PR #8** with partial shmem-shrink + ICA-acknowledgement.
6. **PR #6** as a design RFC (no code change yet).
7. **PR #5** RFC after Kimi K-series release; staged landing (algorithm → carrier → VL).
8. **PR #9** alongside or folded into PR #5 (cuBLAS bypass + cuBLAS-side reproducer issue).
9. **PR #2** folded into PR #5, or as a 6-line follow-up.
10. **PR #12** (engine-agnostic Generator) — now unblocked; file the RFC anytime; code PR follows RFC discussion.
11. **PR #10** if/when PR #8's downstream ICA is resolved (otherwise skip).

PR #4 removed from the order — obsoleted by upstream `627f4a31`.

---

## Phase discovery summary

| Phase | What it surfaced |
|---|---|
| **Phase 2** | Original Block AttnRes algorithm work (174M dense Llama3 paper-Table-1 reproduction). Roots of PR #5. |
| **Phase 4** | Kimi Linear MoE port + AttnRes wrapper. Roots of PR #5's carrier shape. |
| **Phase 11** | Inference + RLHF push. Surfaced **everything else** — PR #1, #2, #3, #4, #6, #7, #8, #9, #10, #11, and the inference-path side of PR #5. (#4 later obsoleted by upstream `627f4a31` on 2026-05-17.) |

The cluster on Phase 11 reflects how integration work (production inference + RL training) is where upstream rough edges surface — the algorithm-stage Phases 2-4 ran inside torchtitan's training path which is more mature.

---

## What goes here vs. `UPSTREAM_PR_LIST.md`

- **`../UPSTREAM_PR_LIST.md`** — long-form inventory + design discussion across all PR candidates. Lives at repo root.
- **`Raising_PRs/<PR>/`** — actionable filing kit for the PR being prepared. Once a PR is filed, add the upstream PR number to that PR's folder; once merged, the folder can be archived or deleted.
