# KDA gate parameters in bf16: a question for #4025

Refs: pytorch/torchtitan#3029, pytorch/torchtitan#4025

Status: **open question, not a claimed defect.** Raised here so it is on the
record before we comment on that PR, and so the decision is not made silently
by whichever tree we happened to copy.

## What each tree does

#4025's `KimiDeltaAttention` declares the gate parameters with no dtype, so
they follow the default:

    self.A_log = nn.Parameter(torch.empty(config.num_heads))
    self.dt_bias = nn.Parameter(torch.empty(config.num_heads, config.head_dim))

and its `kimi_k3_debugmodel` sets `training.dtype="bfloat16"`, so in the
configuration that PR actually runs, both are bf16.

Ours declared them `dtype=torch.float32`. That was not a considered precision
decision on our side either -- the fp32 came from the `empty()` the `uniform_`
init draws into and the parameter inherited it -- and it made the module's
dtypes non-uniform under `training.dtype=bfloat16`, which FSDP2 rejects
outright. We have since changed ours to follow the default dtype, which is a
no-op under the fp32 default our other flavors use.

So neither tree arrived at bf16 gate parameters deliberately. That is the part
worth flagging.

## Why the dtype of these two is not neutral

fla's gate kernel is explicit about precision, and it splits the two
directions:

* Forward is computed in fp32 regardless of storage --
  `b_A = tl.load(A_log + i_h).to(tl.float32)` (`fla/ops/kda/gate.py:107`),
  and the reference path likewise does `-A_log.view(H, 1).float().exp()`.
  So bf16 storage costs nothing on the forward.
* The **gradient is cast back to the parameter's own dtype**:

      dA = A_log.new_empty(NT, H, dtype=torch.float32)      # accumulate fp32
      ...
      dA = dA.sum(0).view_as(A_log).type_as(A_log)          # then downcast

  (`fla/ops/kda/gate.py:254,276`.) The kernel takes care to accumulate in
  fp32 and then hands back whatever precision the parameter is stored at.

Against that, our own gradient measurements (`GRADIENT_METRICS_2026-08-03.md`)
put `A_log`'s gradients at 4.8e-6 to 1.1e-3, against a model median near
2.5e-3 -- two to three orders below. bf16 carries roughly three decimal
digits, so a gradient at 1e-5 is being quantized fairly coarsely on the way
into an fp32 optimizer state.

That is the concern in one sentence: the kernel accumulates these gradients in
fp32 on purpose, and storing the parameter in bf16 discards that at the last
step, for the parameters whose gradients are smallest in the model.

## What this does NOT show

* No measured convergence difference. Nobody has run the ablation, and the
  step-loss horizon here (3-10 steps on a debug model) could not resolve one.
* Under torchtitan's ordinary recipe the question does not arise at all:
  `training.dtype` stays float32 and FSDP's `mixed_precision_param=bfloat16`
  gives bf16 compute with fp32 master weights, so `A_log` is fp32 anyway. It
  only bites when `training.dtype=bfloat16`, which is what #4025's debugmodel
  chooses.
* The decay parametrization being precision-sensitive in linear-attention /
  SSM families is a claim we have not verified against upstream sources here.
  Do not put it in a PR comment as established.

## Proposed comment on #4025

Neutral and answerable:

> `training.dtype="bfloat16"` in `kimi_k3_debugmodel` puts `A_log` and
> `dt_bias` in bf16. fla's gate kernel accumulates their gradients in fp32 and
> then downcasts to the parameter dtype
> (`ops/kda/gate.py`, `dA.sum(0).view_as(A_log).type_as(A_log)`), and these are
> among the smallest gradients in the model. Was fp32 storage for the two gate
> parameters considered? If they should stay fp32, note that FSDP2 needs
> uniform dtype within a group, so it would need them in their own group
> rather than just a dtype on the parameter.

The second sentence is the useful half: it is the reason a naive "just make
them fp32" fix does not work, and we hit it directly.

## If we want to settle it ourselves

The ablation is cheap and the instrument already exists: same flavor, shared
warm checkpoint, one varied dimension (gate-parameter dtype), and
`grad_attrib_probe.py` comparing per-parameter gradient direction. What it
would show is how much of `A_log`'s gradient survives the downcast -- which is
a measurement, unlike everything above the line here.
