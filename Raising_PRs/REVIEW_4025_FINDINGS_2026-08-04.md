# Findings from reading #4025's tree

Refs: pytorch/torchtitan#3029, pytorch/torchtitan#4025

Addendum to `REVIEW_4025_DRAFT.md`. That draft is about interfaces we want.
This file is about things in their tree we think are wrong or worth
questioning, kept separate so the two can be sent independently and so we do
not mix "please expose a hook" with "this looks like a bug".

Standing rule for this file: **#4025 being upstream does not make it right.**
Two of our own changes this week were made to match their tree, and one of
them (the KDA gate dtype) we now think is the wrong direction on the merits.
Record the disagreement rather than inherit it silently.

## Finding 1 (likely blocking for CP): `get_vision_positions` cannot see a sequence shard

`torchtitan/models/common/multimodal.py`. This is in **`models/common/`**, not
in `models/kimi_k3/`, so whatever contract it sets becomes the contract for
every multimodal model upstream. That is what makes it worth raising now
rather than when we file our CP PR.

It hard-fails on two conditions:

```python
if num_runs != num_items:
    raise ValueError("Multimodal misalignment: found {num_runs} contiguous run(s) ...")
...
if run_lengths[i] != n_tokens:
    raise ValueError("placeholder run {i} spans {run_lengths[i]} token(s) but ...")
```

Both assume the caller holds the **whole** sequence. Under context parallelism
neither holds:

* A CP rank whose sequence shard contains none of an image's placeholders sees
  `num_runs < num_items` and raises, even though nothing is misaligned -- the
  image's tokens are simply on its peer.
* An image straddling a shard boundary gives that rank a **truncated** run, so
  `run_lengths[i] != n_tokens` and it raises.

Both are normal states under CP, not corruption. The check as written is a
good one for the non-CP case -- silently scattering a mismatched run really
would corrupt embeddings -- so the ask is not to delete it but to make the
"how many of these tokens are mine" question answerable:

> Would you take a variant that accepts the item's global token count plus this
> rank's shard offset, so a partial or absent run is a valid input rather than
> an error? We have this working in a fork (each CP rank exchanges its local
> sentinel count, then slices the encoder output by the prefix sum) and would
> rather implement it against this helper than route around it.

We hit exactly this: our equivalent path had to gain a per-rank sentinel-count
`all_reduce` and a prefix-sum slice, plus a zero-sentinel branch. The
zero-sentinel branch is where our own deadlock lived
(`CP_MULTIMODAL_HANG_RESOLVED_2026-08-04.md`), which is a fair warning to pass
on: it is easy to get the arithmetic right and still hang, because the rank
with nothing to splice must still issue every collective its peers issue.

## Finding 2 (minor, consistency): placeholder and real image paths derive dtype differently

`models/kimi_k3/model.py`. The real path casts the input to the encoder's
parameter dtype:

```python
pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
```

while `_encode_placeholder_image` builds its tensor from the *activation*
dtype:

```python
pixel_values_NPK = torch.zeros(..., dtype=embeddings_BLD.dtype, ...)
```

Under their own debugmodel (`training.dtype="bfloat16"`) these coincide, so
nothing is broken today. They stop coinciding under the more usual torchtitan
recipe -- fp32 parameters with FSDP `mixed_precision_param=bfloat16` -- where
`patch_embed.weight.dtype` is fp32 and the embedding output is bf16. Then the
two data-parallel ranks in the same step feed different dtypes into the same
module, which is the sort of thing that shows up much later as a confusing
error.

Cheap ask: derive both from the same expression.

Not verified end-to-end -- we have not run their tree under fp32 params with
FSDP mixed precision, so this is "these two lines disagree", not "we saw it
fail".

## Finding 3 (question, not a defect): KDA gate parameters in bf16

Full write-up in `phase13_k3like_48b_posttrain/KDA_GATE_DTYPE_2026-08-04.md`;
short version here so the review file is self-contained.

Their `A_log` and `dt_bias` follow the default dtype, and their debugmodel sets
`training.dtype="bfloat16"`, so both are bf16 in the configuration that PR
runs. fla's gate kernel computes the forward in fp32 regardless
(`b_A = tl.load(A_log + i_h).to(tl.float32)`) but casts the **gradient** back
to the parameter's own dtype:

```python
dA = A_log.new_empty(NT, H, dtype=torch.float32)   # accumulated in fp32
dA = dA.sum(0).view_as(A_log).type_as(A_log)       # then downcast
```

The kernel takes trouble to accumulate in fp32 and then hands back whatever
precision the parameter is stored at. We measure `A_log`'s gradients two to
three orders below the model median, so bf16 storage quantizes them where fp32
would not.

Honest framing for the comment: we have **not** measured a convergence
difference, and under the ordinary recipe (fp32 params, FSDP mixed precision)
the question does not arise at all. So this is "was it considered?", not
"this is wrong". The useful half of the comment is the follow-up: FSDP2 needs
uniform dtype within a group, so "just make them fp32" does not work -- they
would need their own group. We know because we tried the naive version and
FSDP2 rejected it outright.

## Where we agree with them, having checked

* Their `pixel_values is None` handling (`_encode_placeholder_image` +
  `add_zero_valued_dependency`) is the right shape, and we reached the same
  design independently. Ours had the same hazard on a second code path they do
  not have (the CP zero-sentinel branch) -- their helper is what named the
  failure mode for us.
* `add_zero_valued_dependency` itself is worth having in `distributed/fsdp.py`
  rather than per-model. We vendored it verbatim so the rebase is a delete.
