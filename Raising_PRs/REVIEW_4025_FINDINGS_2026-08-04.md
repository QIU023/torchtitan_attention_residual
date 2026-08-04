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

The rule cuts both ways, and Finding 1 is where it cut against us: it was
first written up as a defect in their tree and is not one. Before attributing
anything to this PR, check `git log --diff-filter=A` on the file -- two of the
three things here live in code #4025 only consumes.

## Finding 1 (NOT a defect -- reclassified): `get_vision_positions` assumes the whole sequence

`torchtitan/models/common/multimodal.py`.

**Provenance first, because it changes who this is addressed to.** This helper
is NOT #4025's. It came in with PR #3532 ("[kimi k2_7] add kimi k2_7",
2026-07-29) and is already on main, used by three models: `qwen3_5`,
`kimi_k2_7`, and now `kimi_k3`. #4025 is its third consumer, not its author.

**And it is not a bug.** All three consumers refuse CP outright, and two of
them name this exact reason in the error text:

    qwen3_5/parallelize.py:
      "Context Parallel is not yet supported for Qwen3.5. GatedDeltaNet (75% of
       layers) requires full-sequence allgather, and multimodal CP needs vision
       scatter before CP sharding."

    kimi_k2_7/parallelize.py:
      "Context Parallel is not yet supported for Kimi K2.5: vision scatter
       needs the full sequence before CP would shard it."

So the whole-sequence assumption is deliberate, upstream knows what it costs,
and the guard is there precisely so nobody reaches the helper with a shard.
Nothing is broken today.

An earlier draft of this file framed it as a defect to raise. That framing was
wrong and would have read badly: we would have been reporting a limitation
they had already documented in their own error messages.

## What it actually is: the gap they wrote down, which we have filled

The two conditions the helper enforces --

```python
if num_runs != num_items:      # a CP rank may hold none of an image's placeholders
if run_lengths[i] != n_tokens: # or a run truncated at the shard boundary
```

-- are exactly the two states a CP rank normally sees. That is why CP is
guarded off rather than why the guard is wrong.

**Their intended design differs from ours, and the PR must say so.** Both error
messages describe scattering the vision embeddings *before* CP shards the
sequence. We do the opposite: shard first, then have each CP rank select its
slice of the encoder output by a prefix sum over per-rank sentinel counts
(`_select_cp_shard`). Ours is not a shortcut around their plan -- their plan
needs the vision encode to happen ahead of `prepare_context_parallel_input`,
which runs in the trainer before the model forward, so "scatter before
sharding" is a trainer-side change rather than a model-side one. Worth stating
plainly instead of quietly shipping the other design.

Proposed comment, positioned as filling a stated gap rather than reporting a
fault:

> `qwen3_5` and `kimi_k2_7` both turn CP off with a note that multimodal CP
> needs the vision scatter to happen before CP shards the sequence. We have CP
> working for the multimodal path with the opposite split -- shard first, then
> each CP rank takes its slice of the encoder output via a prefix sum over
> per-rank sentinel counts -- which keeps the change inside the model and out
> of the trainer's input path. That needs `get_vision_positions` to accept a
> shard: the item's global token count plus this rank's offset, so an absent or
> truncated run is valid input rather than a `ValueError`. Would you take that,
> or do you prefer the scatter-before-sharding route? Happy to implement
> either; we would rather build on this helper than route around it.

One warning worth passing on from our own implementation: getting the shard
arithmetic right is not sufficient. A rank whose shard holds no sentinel must
still issue every collective its peers issue -- ours returned early, dropped
the vision tower out of the loss graph, and deadlocked on a missing FSDP
reduce-scatter (`CP_MULTIMODAL_HANG_RESOLVED_2026-08-04.md`). That failure mode
is invisible until it hangs.

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
