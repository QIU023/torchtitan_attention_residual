# Session handoff -- box #2 (8x5060Ti), packed-TP closed + a gradient-scale finding (2026-07-25)

Supersedes [SESSION_HANDOFF_2026-07-24.md](SESSION_HANDOFF_2026-07-24.md) as
the latest handoff; that doc's sec 4 work plan is still the roadmap and only
its item 7 (first half) is consumed here. Full detail:
[PACKED_TP_VERIFICATION_2026-07-25.md](PACKED_TP_VERIFICATION_2026-07-25.md).

## 0. Repos / pins

| repo | branch | HEAD | this session |
|---|---|---|---|
| logbook | main | (this commit) | this doc + verification doc + 4 scripts + 3 PR kits |
| torchtitan | `attention_residual_dev` | `20bd4f3a` | packed-base x NoParallel fix (`45ee67ac`) + FSDP gradient-division fix + 2 new tests |
| verl | `kimi_k3_integration` | `60b185fe` | unchanged |
| sglang | `attention_residual_inference` | `e45f675` | unchanged |

Box #2 is a fresh 8x RTX 5060 Ti 16GB (all 8 healthy; the 07-24 box's dead
GPU 1 is gone with it). Environment rebuilt verbatim from the 07-24 recipe
(sec 2 there) -- no deviations, no new pins.

## 1. Done: 07-24 sec 7 item 1 (TP x packed-MXFP4 base on GPU)

- tp2 / tp2cp2 / tp2pp2 + fsdp2tp2cp2 all train, rank-identical, step-1
  within 1e-4 of the no-parallelism reference. Cell table in the
  verification doc sec 1.
- Getting there required a real fix: `ef0fced4` wired the Colwise/Rowwise
  packed path but left the **MoE shared experts** (styled `NoParallel`, so
  `_tp_style` is never set) taking the non-TP branch, where the dequantized
  weight is plain while NoParallel's prepare_input has already made the
  activation a DTensor -> `aten.mm got mixed torch.Tensor and DTensor`, at
  step 1, in all four cells. Fix + new CPU regression test in the fork;
  non-TP paths bit-identical pre/post. Details: verification doc sec 2.
- The CPU probe could never have caught it (it hand-wires one module via
  `apply_packed_mxfp4_tp`, so `_tp_style` is always set). Pattern worth
  remembering: a module-level parity probe does not cover
  parallelize-time *style dispatch*.

## 2. Landed: FSDP gradient division (was K3-only, now matches upstream)

Measured on box #2, root-caused, and **landed** (fork `20bd4f3a`) once the
scope audit answered the question that gated it: this is a **kimi_k3-only**
defect. Verification doc sec 4 has the evidence table, sec 7 the post-fix
regression.

- The trainer's loss is `local_loss_sum / global_valid_tokens`, so DP
  gradients must be summed. The shared `apply_fsdp` disables FSDP's
  gradient division for exactly this reason; **kimi_k3's private
  `apply_fsdp` did not**, so every FSDP kimi_k3 run before this fix divided
  its gradients by the fsdp mesh size (`dp_shard x cp`).
- Directly measured (not inferred from curves): true `param.grad` norm is
  1/2 at dp1cp2, 1/4 at dp1cp4, 1/2 at dp2, 1/4 at dp4; adding the missing
  call restores all of them to the 1-GPU value to 2e-5.
- Invisible so far because AdamW (and Muon) are scale-invariant -- loss
  curves are unaffected. What it does affect: grad clipping engages
  `dp_shard*cp` times later than configured, and every grad_norm ever
  recorded in this logbook is that factor too small.
- **This corrects GAPS_TO_K3_SFT B9**, which called it cosmetic with a
  sqrt(cp) reconstruction. Factor is cp, applies to dp_shard too, and it is
  the gradient, not the display.
- **Scope**: `llama3`, `deepseek_v3`, `qwen3` and `gpt_oss` all use the shared
  `apply_fsdp_to_decoder`, which makes the call (`fsdp.py:287`). Only
  kimi_k3's private copy of the Llama3 layout omitted it -- so the fix aligns
  us with behaviour upstream already has; it does not change a convention.
- **Post-fix regression**: all 14 cells of `run_cp_tp_3d_matrix.sh` still
  PASS, every step-1 loss unchanged, and multi-step trajectories move only
  where FSDP is active and clipping now engages -- verified example: fsdp8
  step-5 5.80943 (division on) -> 5.81329 (fixed). The `tp2cp2
  4.00261 -> 3.89796` example first cited here was **misattributed**; the fix
  moves that cell by 1.6e-4, see verification doc sec 10. Recorded grad_norms
  from before 2026-07-25 are `dp_shard*cp` too small -- **treat every
  pre-07-25 grad_norm in this logbook as re-based**. Muon capstone unchanged
  (scale-invariant), as predicted.

## 3. Next-box work plan (07-24 sec 4, updated)

Unchanged and still the priority order. Deltas only:

- **item 7: CLOSED, both halves.** The GPU cells are done (sec 1), and the
  upstream PR extractions now have written kits -- `Raising_PRs/PR16` (titan
  moe TP+EP scatter), `PR17` (verl checkpoint interval), `PR18` (verl engine
  CP). All three audited against current upstream and still applicable.
  **None filed** -- filing needs your go-ahead.
- **items 1-4 (48B QLoRA SFT, bf16 LoRA re-run, long-context CP,
  full-param) still need bigger hardware** -- 16GB cards change nothing
  there. The packed-MXFP4 path they depend on now composes with TP, which
  was the last missing axis for 48B QLoRA on small VRAM.
- **item 5 (16-rank 5D)** unchanged: physically impossible on 8 cards.
- **grad clipping now behaves as configured** (sec 2), so long runs no longer
  carry the silently-loosened threshold.

## 3b. Also swept this session (everything <=8 GPUs that was still open)

Detail in verification doc sec 8. One line each:

| item | result |
|---|---|
| DCP for the packed-MXFP4 ckpt under TP (`ef0fced4`'s untested claim) | **VERIFIED**: mid-run save + same-mesh resume is bit-exact; cross-mesh reshard to tp1 / fsdp2 / tp2cp2 agrees within 8e-5..2.7e-3 |
| AC full x packed-TP | **PASS**, bit-identical to non-AC tp2 |
| compile x packed-TP | **DOES NOT WORK** -- two independent blockers, both root-caused (sec 8c). Non-TP packed compile is fine; compile x TP is fine for full-param |
| GAPS sec 5 "LoRA without FSDP crashes" | **STALE** -- trains fine; A2 fixed it the day it was filed. Corrected in both docs |
| B7 Muon x FSDP x TP x CP | **PASS**, unchanged post grad-fix (scale-invariant) |
| B8 deltas-compose capstone | **PASS** (0.3584 -> 0.3583) |
| C11 torchao weight-only MXFP4 | **still absent, and 0.17.0 is the latest release** -- nothing to bump to. `NVFP4WeightOnlyConfig` exists but K3 is MXFP4-native |

New finding worth carrying: **every upstream torchtitan model that builds
attention masks is unrunnable in this fork on torch 2.12 stable** --
`create_block_mask() got an unexpected keyword argument
'separate_full_blocks'` (the upstream-merged flex path needs a nightly), and
even shimmed past that, the flex triton kernel wants 139776 B of shared memory
against sm_120's 101376 B limit. kimi_k3 is causal-only and never calls
`get_attention_masks`, which is why no earlier session hit this. It is also
why PR16's upstream-model reproducer could not be produced here.

## 4. Reproduction warning (bit us this session)

The 07-24 handoff quotes `7.5695` as the QLoRA(mxfp4-packed) fsdp2
baseline. It is **not reproducible** without that session's bf16 source
checkpoint (checkpoints are not committed): the packed base values, and so
the absolute loss, depend on which bf16 run was quantized. Rebuilt
artifacts in-session and used a same-session `ref_1gpu` anchor instead --
recipe in verification doc sec 5. Going forward: quote baselines with the
artifact that produced them, or quote only same-session deltas.

Second papercut, same class: `global_batch_size` defaults to
`local_batch_size * dp_degree`, so comparing a dp1 cell against a dp2 cell
without pinning `--training.global-batch-size` compares two different
optimization problems. Every cell in the new matrix pins it.

## 4b. Two pre-commit traps in the fork (matters for the item-8 PR extraction)

Both cost a revert this session; the fork commit was made with hooks
bypassed and the hooks verified by hand instead.

- **`pyrefly` must stay unpinned-free.** Installing `pyrefly` from PyPI and
  running the hook strips `# pyrefly: ignore [...]` comments across ~20
  unrelated core files (local version disagrees with the pinned one about
  which suppressions are needed). Never commit that sweep.
- **`kimi_k3/lora.py` is not ufmt-clean at HEAD**, so any `pre-commit run`
  reformats ~25 pre-existing lines in it. Keep functional commits clean of
  that churn, but the upstream PR extraction WILL have to run
  `pre-commit run --all-files` and land the reformat as its own commit.

## 5. Honesty carries (unchanged)

- Never claim 2.8T personally validated; 48B real weights + K3-faithful
  topology is the ceiling.
- Debug-scale cells verify MECHANISMS, not training quality.
- "Loss descends within a band" is not a multi-rank correctness gate.
  New corollary from sec 2: rank-identical grad_norm is a cross-rank
  consistency gate, not a magnitude gate -- a uniformly mis-scaled gradient
  is rank-identical and Adam-invisible.
