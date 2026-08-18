# PR27 -- torchtitan: let GroupedExperts subclasses change the GLU variant without copying forward

**Target**: `pytorch/torchtitan`, `torchtitan/models/common/moe.py` only.
**Independent of PR-4025** -- it touches no model folder, and both the K3 PRs and
gpt_oss benefit.

**Status**: not filed. Patch: `gate_up_combine.patch` (applies to upstream `main`).

## PASTE (PR description)

`GroupedExperts.forward` computes `F.silu(w1(x)) * w3(x)` inline. This moves that one
line into a `gate_up_combine(gate, up)` method that subclasses can override. Default is
SwiGLU, unchanged, so no existing model moves.

A hook rather than an `activation=` argument, because the variants that exist are not
single-argument activations. gpt_oss clamps both branches at a configured limit; Kimi
K3's SiTU-GLU is `beta * tanh(g / beta) * sigmoid(g)` with a second clip on the linear
branch, computed in fp32 because the product of two saturating nonlinearities is
sensitive to bf16 rounding near the caps. Neither is `act(gate) * up`, and both carry
hyperparameters the subclass owns.

What this removes is a real copy. Our K3 expert class was 92 lines reproducing the whole
forward -- DTensor unwrapping, offset cumsum, SPMD type mutation, all three grouped-mm
calls -- to change that one step. It is 53 lines now and overrides only the hook. The
copy was also a quantization hazard: the MXFP8 converter installs its GEMM by overriding
`_grouped_mm`, so a copied forward calling `torch._grouped_mm` directly silently opts
every routed expert out of it, and nothing raises.

The two grouped-mms are named `gate_RF` / `up_RF` rather than being folded into one
expression, which is why the diff looks larger than one line.

## Evidence

- A stock (unmodified) MoE model trains to identical loss and grad_norm before and after
  -- the default path is the same arithmetic in the same order, not merely equivalent.
  See `RESULTS.md`.
- On the subclassing side, the K3 three-arm parallelism matrix passes on the hook version
  including every EP cell, which is where the `_grouped_mm` seam matters.

## Note for the reviewer

`MoE.forward` in the same file takes a second, independent change in PR28
(`router_input_BLD`). The two patches apply to `main` in either order; they do not
overlap.
