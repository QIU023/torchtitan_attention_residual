> **SUPERSEDED as the primary route, 2026-08-18.** For a 13-line change with the
> maintainer already inviting it, an issue plus a PR is process for its own sake. File
> ONE PR (`PR_pytorch.md`) and answer her two design questions in a reply on the
> torchtitan thread, where she asked them. This file stays in case she asks for the
> design to be split out into its own issue -- the Q&A below is still the answer.

# pytorch/pytorch issue: a dtype argument for `get_total_norm`

Invited by janeyx99 on the torchtitan PR (2026-08-17): *"I'm more confident this can be
resolved in pytorch/pytorch if you open an issue/PR and tag me!"* -- after two design
questions, both answered in the body below because she asked them.

**Target**: `pytorch/pytorch`, `torch/nn/utils/clip_grad.py`
**Tag**: @janeyx99
**Do NOT file without a human's go-ahead.**

## Verified against pytorch/main before drafting

| claim in the body | how it was checked |
|---|---|
| the public symbol is `torch.nn.utils.get_total_norm` | `torch/nn/utils/__init__.py` exports `_get_total_norm as get_total_norm`; `__all__` in `clip_grad.py` does not list it |
| the three norm call sites are as quoted | read `_get_total_norm` verbatim on `main` |
| `clip_grad_norm_` also scales | it calls `_clip_grads_with_norm_`, which does `torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))` |
| `vector_norm` documents `dtype` as the pre-accumulation cast | its docstring: "If specified :attr:`x` is cast to :attr:`dtype` prior to doing the accumulation" |
| `_foreach_norm` accepts `dtype`, including an explicit `None` | measured on torch 2.8: omitted and `dtype=None` both return bfloat16, `dtype=torch.float32` returns float32 |

The patch and its probe are in this folder: `get_total_norm_dtype_pytorch.patch`
(against `main`, `git apply --check` clean) and `probe_get_total_norm_dtype.py`, which
measures both arms and asserts the patched one.

## Title

    torch.nn.utils.get_total_norm: add a dtype argument for the accumulation

## Body

--- PASTE BEGIN ---

`get_total_norm` accumulates in the input tensors' dtype. With bf16 tensors both the
per-tensor norms and the norm-of-norms are bf16, and since the rounding happens per group
the result depends on how the tensors were grouped rather than only on their values. 512 bf16
gradients, seed 0, the same tensors at every world size; the only thing that changes is how
they are split across ranks, and the cross-rank reduce is the real one (nccl), not a float64
emulation. Two splits that both occur in training -- FSDP (each gradient a DTensor `Shard(0)`,
combined via `_NormPartial` + `full_tensor()`) and pipeline (rank r owns `grads[r::world]`,
partials all-reduced):

```
                world=1    world=2    world=4    world=8    spread(rel)
dtensor today   256.000    256.000    256.000    256.000    0.00e+00
dtensor fp32    255.682    255.682    255.682    255.682    0.00e+00
pp today        256.000    255.973    255.505    255.631    1.93e-03
pp fp32         255.682    255.682    255.682    255.682    5.97e-08
float64 truth   255.682226
```

The pipeline split is world-dependent -- 256.000 at world 1, 255.505 at world 4, relative
spread 1.93e-3 -- so the same gradients report a different norm depending only on the number
of stages. The DTensor split is world-independent but bf16-imprecise (256.000 against the
truth 255.682). With the argument set to `torch.float32` both splits reduce to the float64
truth and the spread collapses to 6e-8.

Under pipeline parallelism that partition is the model cut. Two torchtitan layouts of one
model with bit-identical gradients (788 shard norms, max relative difference 0.000e+00)
reported grad_norm 10.008054 and 9.951641 against a true 9.989287; with max_norm=1.0
clipping fires every step, so the two layouts took steps differing by 0.566% from the same
gradients.

Proposal: `dtype: torch.dtype | None = None`, passed to the calls the function already
makes.

```python
norms.extend(torch._foreach_norm(device_tensors, norm_type, dtype=dtype))
norms.extend([torch.linalg.vector_norm(g, norm_type, dtype=dtype) for g in device_tensors])
total_norm = torch.linalg.vector_norm(torch.stack([...]), norm_type, dtype=dtype)
```

No signature change to `vector_norm` or `_foreach_norm`: both already take `dtype`, and
`vector_norm` documents it as the cast done "prior to doing the accumulation", so the name
and the meaning carry over instead of being invented here.

One dependency: DTensor gradients (the FSDP case) reach `torch._foreach_norm(..., dtype=)`,
and today that raises -- `_foreach_norm.Scalar` shares `vector_norm`'s DTensor sharding
strategy, which reads `args_schema[2]` as `dim`, but `_foreach_norm`'s position 2 is `dtype`
(its schema has no dim), so `normalize_dim` gets a dtype and errors. The one-line dispatch
fix (set `dim=None` for the `_foreach_norm.Scalar` overload) lets `get_total_norm` stay a
pure passthrough with no DTensor special-casing, and fixes every `_foreach_norm(dtype=)`
DTensor caller rather than only this function. It is a separate commit
(`dtensor_foreach_norm_dtype_pytorch.patch`); this PR carries both. Verified together at
world 1/2/4/8 on CUDA/nccl (torch 2.14): both the sharded (DTensor) and pipeline splits
reduce to the float64 truth, and without the dispatch fix the DTensor arm crashes.

On the ambiguity question -- `get_total_norm` computes nothing but those norms, so one
argument covers all of it. It would be ambiguous one level up on `clip_grad_norm_`, which
also scales the gradients through `_clip_grads_with_norm_`; a caller who wants fp32 there
already has `get_total_norm(..., dtype=torch.float32)` followed by
`clip_grads_with_norm_(...)`. So the argument belongs on `get_total_norm` only.

`dtype=None` is current behaviour exactly. The one visible change when it is set is that
the returned norm is in `dtype` rather than the inputs', matching `vector_norm`.

Happy to send the PR -- the patch and a probe that measures both arms are written.
torchtitan carries a private copy of this function today to get the fp32 reduction
(pytorch/torchtitan#4135); with this argument it goes back to a one-line call.

--- PASTE END ---

## Reply to post on the torchtitan PR, after the issue exists

--- PASTE BEGIN ---

Opened pytorch/pytorch#NNNNN and tagged you.

Short answers to both: no API change to `vector_norm` or `_foreach_norm` -- they already
take `dtype`, which is why the diff here is only the norm calls. And the ambiguity does not
arise inside `get_total_norm`, which computes nothing but norms; it would on
`clip_grad_norm_`, which also scales, so the argument goes on `get_total_norm` only. Once
it lands, the helper in this PR becomes a one-line call.

--- PASTE END ---

## Held back unless asked

* The provenance: the two layouts were a vision tower on its own pipeline stage vs inline,
  and the gradients were proven identical before the norm was suspected
  (`phase13_k3like_48b_posttrain/DEP_ALIGNMENT_2026-08-09.md`).
* The training-level table (llama3_debugmodel, dp_shard=2 x pp=2: 1.4453 bf16 / 1.4508
  patched / 1.4509 fp32). It is torchtitan-level evidence and this is a torch-level issue.
* Cost: the per-tensor norms are already memory-bound reads; promoting the accumulator does
  not change read volume, and the norm-of-norms is over one scalar per tensor.

## What this does not settle

Whether torchtitan should then default to fp32 or expose it as a config field. That is the
downstream decision and stays on the torchtitan PR -- the argument only makes it expressible.
