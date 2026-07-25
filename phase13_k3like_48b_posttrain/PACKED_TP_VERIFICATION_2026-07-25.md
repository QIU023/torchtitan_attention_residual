# packed-MXFP4 QLoRA x TP verification (2026-07-25, 8x RTX 5060 Ti 16GB, box #2)

Closes [SESSION_HANDOFF_2026-07-24.md](SESSION_HANDOFF_2026-07-24.md) sec 7
item 1: the GPU cells for TP x packed-MXFP4 base (fork `ef0fced4` wired it;
only the CPU-gloo probe had run before the previous box lost GPU 1).

Reproduce with [run_packed_tp_matrix.sh](run_packed_tp_matrix.sh).
Box #2 is the same class as the dead one (8x 5060 Ti 16GB, cc 12.0, driver
580.173.02, CUDA 13.0); all 8 GPUs healthy. Environment rebuilt exactly per
the 07-24 handoff sec 2 -- torch 2.12.0+cu130, fla-core 0.5.1, torchao
0.17.0, transformers 5.14.1. Gates re-run before any new cell: unit suite
**84 passed + 66 subtests** (identical to the 07-24 snapshot);
`packed_tp_cpu_probe.py` colwise+rowwise fwd 4.77e-7, adapter grads
<= 1.55e-6 (matches the 5e-7 / 2e-6 recorded on the old box).

## 0. Headline

1. **All three TP cells crashed on first run** -- `ef0fced4`'s packed-TP
   wiring was incomplete. Root cause found and fixed (sec 2); the fix is a
   provable no-op on every non-TP path.
2. With the fix, tp2 / tp2cp2 / tp2pp2 all train with **rank-identical**
   loss+grad_norm and step-1 within **1e-4** of the no-parallelism
   reference -- two orders tighter than the TP band the 07-24 doc records
   for the full-param flavor (4.6e-3).
3. Separately, and NOT specific to packed-MXFP4: **the applied gradient is
   under-scaled by the FSDP mesh size (dp_shard x cp)** in every FSDP
   kimi_k3 run. Measured directly, root-caused, one-line fix identified,
   NOT landed (sec 4) -- it changes effective LR/clipping semantics for
   every run in the logbook, so it is the user's call.

## 1. The matrix (5-step cells, seed 42 deterministic, bf16, seq 512)

`kimi_linear_debugmodel_gated_qlora_mxfp4`, packed base loaded from an
offline-quantized DCP checkpoint. **`--training.global-batch-size 4` is
pinned on every cell**: `global_batch_size` defaults to `local_batch_size *
dp_degree`, so a dp2 cell silently optimizes a 2x larger batch than a dp1
cell and the two losses are not comparable. `ref_1gpu` (no parallelism at
all) is the anchor for the dp1 TP cells.

| cell | ranks | step-1 | step-5 | grad_norm@1 | vs ref_1gpu | rank-identical |
|---|---|---|---|---|---|---|
| **ref_1gpu** (dp1, no TP/CP/PP) | 1 | 7.58917 | 7.57660 | 0.1672 | — | yes |
| **tp2** | 2 | 7.58910 | 7.57648 | 0.1672 | **7e-5** | yes |
| **tp2cp2** | 4 | 7.58911 | 7.57662 | 0.0835 | **6e-5** | yes |
| **tp2pp2** (1F1B) | 4 | 7.58918 | 7.57659 | 0.1671 | **1e-5** | yes |
| fsdp2 (the 07-24 baseline shape) | 2 | 7.58189 | 7.58304 | 0.0886 | n/a (different batch split) | yes |
| fsdp2tp2cp2 | 8 | 7.58193 | 7.58297 | 0.0442 | n/a | yes |

Peak memory: 0.23-0.24 GiB (tp2 / fsdp2), 0.14 GiB (fsdp2tp2cp2).

Notes on reading this table:

- The tp2cp2 / fsdp2tp2cp2 grad_norm columns are *not* rank divergence and
  *not* a TP effect -- they are the FSDP-mesh gradient division of sec 4
  (cp2 halves it, dp2xcp2 quarters it). Loss agreement is what carries the
  TP claim here.
- fsdp2 vs ref_1gpu differ by 7e-3 because at dp2 the same global batch is
  split across ranks instead of across 2 grad-accum microbatches, and dp2
  additionally runs under FSDP mixed precision. Not comparable by
  construction; the dp1 cells are.
- `tp2pp2` logs an extra `loss: -2.00000` line from the stage that does not
  own the loss -- the PP sentinel. `run_packed_tp_matrix.sh` filters
  `-2.00000|-4.00000` exactly as `run_cp_tp_3d_matrix.sh` does; an earlier
  revision of the script did not, and reported a phantom "10 unique lines"
  for that cell.

**Why the parity is so much tighter than the full-param TP band.** Colwise
shards the packed base on rows and MX block-32 quantization is
row-blockwise, so each rank dequantizes a bit-exact row slice; rowwise
shards on whole 32-blocks. TP therefore adds no quantization error on top,
only matmul reduction order -- and the frozen base means most of the model
contributes no gradient path at all.

## 2. The bug `ef0fced4` shipped: packed base x NoParallel

Every TP cell died at step 1:

```
File ".../experiments/kimi_k3/lora.py", line 368, in forward
    base_out = F.linear(x, w)
RuntimeError: aten.mm.default got mixed torch.Tensor and DTensor
```
via `moe.py:501 self.shared_experts(x_BLD)` -> `feed_forward.py:54 self.w1(x)`.

`ef0fced4` handles the packed base for the Colwise/Rowwise-styled
projections (`_tp_style` set -> `_forward_packed_tp`). But the **MoE shared
experts** (`w1/w2/w3`) are styled `NoParallel`, not Colwise/Rowwise, so:

- the `packed_tp` loop in `apply_tp_kimi_linear` never sees them and
  `_tp_style` stays `None` -> forward takes the non-TP branch;
- that branch calls `_dequant_base_mxfp4()`, which for `_tp_style is None`
  densifies `base_qdata`/`base_scale` via `full_tensor()` -> a **plain**
  weight;
- meanwhile `ef0fced4` promoted those same packed params to
  `DTensor(Replicate)` ("keeping every param in the FSDP unit on one
  mesh"), and `NoParallel._prepare_input_fn` wraps the module input into a
  **DTensor** -> `F.linear(DTensor, plain)` -> mixed-operand error.

The bf16 branch has the symmetric guard already (`lora.py:372-383`, "Plain
input but a DTensor base weight ... densify to match"); the mxfp4 branch
covers neither direction. `packed_tp_cpu_probe.py` cannot catch this: it
wires one `KimiLoRALinear` by hand via `apply_packed_mxfp4_tp`, so it only
ever exercises the `_tp_style`-set path.

**Fix** (fork, `lora.py`), mirroring the adapter alignment 20 lines below it:

```python
w = self._dequant_base_mxfp4().to(x.dtype)
if x_is_dt:
    mesh = x.device_mesh
    w = DTensor.from_local(w, mesh, [Replicate()] * mesh.ndim, run_check=False)
base_out = F.linear(x, w)
```

Replicated compute is the intent `ef0fced4` declared for non-styled packed
modules; this aligns the forward with it rather than changing the design.

Evidence the fix is scoped:

- **non-TP paths bit-identical.** The `fsdp2` cell's full 5-step
  loss+grad_norm sequence (7.58189 / 7.59986 / 7.61904 / 7.58710 / 7.58304,
  norms 0.0886 / 0.0950 / 0.0872 / 0.1087 / 0.1335) is identical pre- and
  post-fix -- the new branch is guarded on `x_is_dt`, which is False
  without TP.
- unit suite still 84 passed + 66 subtests, plus a new CPU test
  (`test_cp_qlora_fixes.py::TestPackedMXFP4NoParallelInput`) that drives a
  packed module with a DTensor input on a 1-rank gloo mesh and asserts
  equality with the plain-input result. Verified to FAIL with the exact
  production error when the fix is reverted.

## 3. What sec 7 item 1 asked for: done

tp2, tp2cp2, tp2pp2 for the packed-MXFP4 base all run and are
parity-checked. `fsdp2tp2cp2` (packed base x FSDP x TP x CP, 8 ranks) is
added as the composition cell. The `TP x packed-MXFP4 base` entry in
GAPS_TO_K3_SFT / the 07-24 doc's "Deferred by choice" list can be closed.

## 4. Separate finding: FSDP under-scales the applied gradient by (dp_shard x cp)

`torchtitan.trainer` computes the loss as `local_loss_sum /
global_valid_tokens` (trainer.py:762), so gradients must be **summed**
across data-parallel ranks. The shared
`torchtitan/distributed/fsdp.py::apply_fsdp` does exactly that -- line 287
calls `disable_fsdp_gradient_division(model)` ("We handle gradient scaling
ourselves in the training loop with global token count"). **kimi_k3's
private `apply_fsdp` (parallelize.py:971) never calls it**, so FSDP2's
default mean-reduction divides every gradient by the fsdp mesh size.

Measured directly, not inferred: `cp_grad_scale_probe.py` wraps
`clip_grad_norm_`, densifies each `param.grad` with `full_tensor()` and sums
squares by hand, bypassing `get_total_norm`/`_NormPartial` entirely. Same 57
grad tensors in every run:

| cell | fsdp mesh size | as shipped | with the one-line fix | ratio |
|---|---|---|---|---|
| dp1cp1 (no FSDP at all) | — | 0.167192 | 0.167192 | 1.000 |
| dp1cp2 | 2 | 0.083585 | 0.167170 | **2.000** |
| dp1cp4 | 4 | 0.041790 | 0.167161 | **4.000** |
| dp2cp1 (global batch 4) | 2 | 0.088589 | 0.177177 | **2.000** |
| dp4cp1 (global batch 8) | 4 | 0.037435 | 0.149739 | **4.000** |

The hand-computed norms match the printed metric to 4-5 digits, so the
printed grad_norm is the true global norm -- **not** a local partial.

**This corrects GAPS_TO_K3_SFT B9**, which called the CP grad_norm drop
cosmetic ("the metric prints the local partial norm ... clipping is
globally consistent") and offered a sqrt(cp) reconstruction. Both are
wrong: the factor is exactly cp (not sqrt(cp)), it applies to dp_shard as
well as cp, and it is the applied gradient rather than the display.

**Why every 07-24 cell still passed.** AdamW is scale-invariant to a
uniform gradient scaling (`m/sqrt(v)` cancels it), so loss trajectories are
unaffected -- tp2cp2 tracks ref_1gpu to 1e-4 while carrying a 2x-scaled
gradient. Muon's Newton-Schulz orthogonalization is likewise
scale-invariant. Demonstrated by breaking the invariance: with
`eps=1.0, lr=0.5` and clipping off (update ~ linear in grad), cp1 vs cp2
diverge monotonically (step-2 1.2e-3, step-3 4.8e-3, step-4 8.3e-3), and
doubling cp2's lr moves it ~65% of the way back (the residual is expected:
eps=1.0 does not fully dominate sqrt(v), so the compensation is not exactly
linear).

**What it actually breaks** (nothing already recorded, hence "finding" not
"retraction"):

- **grad clipping**: `max_norm=1.0` behaves as `dp_shard*cp x 1.0`, so
  clipping engages that much later. Any run that relies on clipping (long
  SFT, spiky data) diverges from its 1-GPU equivalent.
- **every recorded grad_norm** in the logbook is `1/(dp_shard*cp)` of the
  true value -- including "rank-identical grad_norm", which stays a valid
  cross-rank consistency gate but is not a magnitude gate.
- any scale-sensitive optimizer or loss-scaling path added later.

**Proposed fix** (one line, matches upstream), at the end of kimi_k3's
`apply_fsdp`:

```python
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
disable_fsdp_gradient_division(model)
```

**NOT landed.** It changes the effective LR and clipping semantics of every
FSDP kimi_k3 run and re-bases every grad_norm in the logbook, which is a
scope call for the user, not a side effect of the packed-TP task. The probe
script is committed so the decision costs one command.

## 5. Reproduction recipe (artifacts are not portable)

The packed base values depend on the bf16 source checkpoint, so **absolute
step-1 losses are only comparable within one packed artifact**. The 07-24
handoff's `7.5695` is not reproducible here: that number came from that
session's own bf16 source run, which was not committed (checkpoints are not
in the repo). Rebuilt in-session:

```bash
# 1. bf16 source (dp2, 1 step, model-only fp32 save at last step)
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 \
  --config kimi_linear_debugmodel_gated_lora --debug.seed 42 \
  --debug.deterministic --training.steps 1 \
  --parallelism.data_parallel_shard_degree 2 \
  --checkpoint.enable --checkpoint.interval 1 --dump-folder /workspace/out_bf16_lora
# 2. offline streaming quantize -> packed layout  (packs 15 bases, copies 130)
python phase13_k3like_48b_posttrain/stream_quantize_mxfp4_dcp.py \
  --src /workspace/out_bf16_lora/checkpoint/step-1 --dst /workspace/packed_mxfp4_ckpt
# 3. the matrix
STEPS=5 GLOBAL_BATCH=4 bash phase13_k3like_48b_posttrain/run_packed_tp_matrix.sh
```

Lesson for future baselines: quote a baseline together with the artifact
that produced it, or quote only deltas measured inside one session.

## 6. Honesty carries

- Debug-scale cells verify MECHANISMS, not training quality.
- Rank-identical loss+grad_norm is a cross-rank consistency gate only; sec
  4 is the counterexample to treating it as a magnitude gate.
- 48B real weights, 16-rank 5D, and long-context remain not run here.
