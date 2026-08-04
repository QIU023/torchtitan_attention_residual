# compile + EP: `_grouped_mm` cannot take a zero-length contraction dim

Refs: pytorch/torchtitan#3029

Supersedes the attribution in `COMPILE_EP_GROUPED_MM_2026-08-04.md`, which
concluded "the defect is in this fork, prime suspect is the routing-map
change". That was wrong. The routing-map change is not involved.

## Minimal reproduction

No model, no torch.compile, no distributed, five lines:

```python
a = torch.empty(224, 0, device="cuda", dtype=torch.bfloat16)   # K = 0
w = torch.randn(4, 0, 256, device="cuda", dtype=torch.bfloat16)
offs = torch.zeros(4, device="cuda", dtype=torch.int32)
torch._grouped_mm(a, w, offs=offs)
```

    contiguous, strides (1,1)  -> RuntimeError: strides should be multiple of 16 bytes
    strides (0,1)              -> RuntimeError: Invalid strides/sizes, got [0, 1]
                                  for strides and [224, 0] for sizes

The second message is byte-for-byte the one the compiled EP legs die with.

## What is and is not tolerated

Tested directly rather than assumed. `_grouped_mm` handles empty **groups**
perfectly well, eager and compiled, including a rank with no tokens at all:

| group sizes | R | eager | compiled |
|---|---|---|---|
| [128,128,128,128] | 512 | ok | ok |
| [256,256,0,0] | 512 | ok | ok |
| [512,0,0,0] | 512 | ok | ok |
| [0,256,0,256] | 512 | ok | ok |
| [0,0,0,0] | 0 | ok | ok |

So "an expert got zero tokens" is not the failure, and the earlier note that
`_grouped_mm` "is intolerant of degenerate shapes" was too broad -- it is
intolerant of exactly one thing, an operand whose **contraction** dimension is
zero.

That shape does not arise in the forward. It arises in the weight-gradient
form, `h_e^T @ grad_out_e`, where the contraction dimension is the token count.

## Why eager training passes and compiled does not

The traceback puts the failure in `_compiled_backward` ->
`extern_kernels._grouped_mm(buf2, primals_8, cumsum)`, i.e. inductor's
generated backward. Eager's autograd for this op does not emit that call when
there is nothing to contract; inductor emits it unconditionally.

So compile is not miscompiling anything. It is issuing a call that the
operator cannot service, in a case the eager path avoids.

## Where the empty groups come from, measured

Per-rank instrumentation of every `_grouped_mm` call in the failing
configuration (twin flavor, dp2 x ep2 x tp2 x pp2, eager):

    rank0: a=[248,256] b=[4,256,224] group_sizes=[9,24,188,27]   empty_groups=0
    rank4: a=[512,256] b=[4,256,224] group_sizes=[256,256,0,0]   empty_groups=2

Empty groups are routine here and eager runs fine with them. At 8 experts top-2
on a 256-token sequence they are unavoidable, and at any real EP scale they are
the normal case rather than the exception.

## Consequences

* **Not ours.** Nothing about our routing-map scatter or LatentMoE is implicated.
  The earlier control (deepseek_v3 compiled+EP passing on #4025's tree) is still
  a true observation, but it does not mean what it was taken to mean: that model
  simply may not produce a zero-token contraction in backward.
* **Reportable upstream** (pytorch/pytorch, not torchtitan): `_grouped_mm`'s
  compiled backward fails whenever a group is empty and the weight gradient is
  taken, which is a normal MoE state. Either the op should accept a zero-length
  contraction dim (returning zeros) or inductor should skip the call.
* **Nothing published depends on it.** #4025 scopes `torch.compile` out
  explicitly, and every matrix we report is compile-off. The compiled twin
  matrix is 10/13, and the three failures are all EP legs with this single
  cause.

## Not fixed here

A workaround exists -- guard the call when the contraction dim is zero -- but
it belongs upstream in the operator or in inductor's lowering, not as a
per-model patch in `models/common/moe.py`. Recording the diagnosis and the
five-line reproduction so the upstream report can be filed with both.
