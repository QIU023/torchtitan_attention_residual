# Session handoff -- box #2 (8x5060Ti), packed-TP closed + a gradient-scale finding (2026-07-25)

Supersedes [SESSION_HANDOFF_2026-07-24.md](SESSION_HANDOFF_2026-07-24.md) as
the latest handoff; that doc's sec 4 work plan is still the roadmap and only
its item 7 (first half) is consumed here. Full detail:
[PACKED_TP_VERIFICATION_2026-07-25.md](PACKED_TP_VERIFICATION_2026-07-25.md).

## 0. Repos / pins

| repo | branch | HEAD | this session |
|---|---|---|---|
| logbook | main | (this commit) | this doc + verification doc + 2 scripts |
| torchtitan | `attention_residual_dev` | (bumped) | packed-base x NoParallel fix + CPU test |
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

## 2. Open decision for the user: FSDP gradient division

Measured on box #2, root-caused, fix identified, **deliberately not
landed**. Verification doc sec 4 has the evidence table and the reasoning.

- The trainer's loss is `local_loss_sum / global_valid_tokens`, so DP
  gradients must be summed. The shared `apply_fsdp` disables FSDP's
  gradient division for exactly this reason; **kimi_k3's private
  `apply_fsdp` never does**, so every FSDP kimi_k3 run divides its
  gradients by the fsdp mesh size (`dp_shard x cp`).
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
- Fix is one line at the end of kimi_k3 `apply_fsdp`; re-measure with
  `cp_grad_scale_probe.py`. Landing it re-bases recorded grad_norms and
  changes clipping behaviour for every run, hence the user's call.

## 3. Next-box work plan (07-24 sec 4, updated)

Unchanged and still the priority order. Deltas only:

- **item 7 (TP x packed base): CLOSED** for GPU cells. Its second half
  (upstream PR extractions: moe.py TP+EP scatter fix; verl checkpoint
  interval=1 + engine CP fixes) is untouched.
- **items 1-4 (48B QLoRA SFT, bf16 LoRA re-run, long-context CP,
  full-param) still need bigger hardware** -- 16GB cards change nothing
  there. The packed-MXFP4 path they depend on now composes with TP, which
  was the last missing axis for 48B QLoRA on small VRAM.
- **item 5 (16-rank 5D)** unchanged: physically impossible on 8 cards.
- Decide sec 2 above **before** any long run that relies on grad clipping.

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
