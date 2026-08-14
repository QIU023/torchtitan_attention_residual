# Every change we make to shared torchtitan files, and whether it survives

Nine core files carry our diff. A reviewer sees these before they see our model, so each
one needs an answer to a single question: **can an upstream model reach the same failure?**
If yes it is a standalone bug-fix PR. If no it does not belong in a shared file.

The two flagged in the handoff as unjustified were both checked by measurement rather than
by re-reading the comment that justified them, and both fell.

| file | change | verdict |
|---|---|---|
| `distributed/pipeline_parallel.py` | divide `local_batch_size` by the PP microbatch size | **REMOVED** -- premise false |
| `models/common/moe.py` | DTensor branch in the routing-map scatter | **REMOVED** -- premise false |
| `models/common/moe.py` | `router_input_BLD` keyword | KEEP, dissolves at migration step 3 |
| `models/utils.py` | skip non-Shard placements before reading `.dim` | KEEP -- clean upstream bug, file it |
| `distributed/fsdp.py` | `add_zero_valued_dependency` | KEEP -- vendored from the upstream K3 PR |
| `distributed/utils.py` | `_get_total_norm_fp32` | KEEP -- already its own PR, ready to file |
| `components/optimizer.py`, `components/lr_scheduler.py` | allow a stage with no trainable params | KEEP -- wants its own PR |
| `tools/grouped_mm_empty_shim.py` | empty-contraction shim | KEEP -- upstream PyTorch PR pending |
| `models/__init__.py` | one registry line | KEEP |

## The two that fell

### `pipeline_parallel.py`: the divide

The change read:

```python
local_batch_size=(
    training.local_batch_size // parallelism.pipeline_parallel_microbatch_size
),
```

justified by "since the dataloader change the loader emits ONE pipeline microbatch per
fetch, so passing the local batch counts the split twice".

The first half is true and the conclusion does not follow. Upstream is already consistent
about it, in three places that agree:

* `trainer.py` builds the dataloader at `pipeline_parallel_microbatch_size` when PP is on;
* `trainer.py` sets `num_pipeline_parallel_microbatches = local_batch_size // microbatch_size`
  and collects exactly that many fetches into `arg_mbs`;
* `_build_pipeline_schedule` computes `n_microbatches = local_batch_size // microbatch_size`
  from the undivided value it is passed.

`len(arg_mbs) == n_microbatches` on both sides. Our extra division makes it
`local_batch_size // microbatch_size**2`, which is a no-op at the default microbatch size
of 1 -- the only value anything in our matrix uses -- and wrong above it. It was inert
where we tested and incorrect where we did not.

It also entered the tree inside a commit titled "kimi_k3: compute alpha in FP32", which is
how it escaped review: nothing in the message mentions PP.

### `common/moe.py`: the scatter branch

The change replaced upstream's one-line scatter with a DTensor branch, justified by
"DTensor has no scatter strategy that preserves Shard(dim=1) -- in-place `scatter_` errors,
out-of-place `scatter` would redistribute to Replicate".

Measured directly, on gloo/CPU with no model and no GPU
(`matrix_scripts/` sibling of this doc, two ranks):

| scores | index | `scatter_` | resulting placement |
|---|---|---|---|
| `Shard(1)` | `Shard(1)` | OK | `Shard(1)` |
| `Shard(1)` | `Replicate` | OK | `Shard(1)` |
| `Replicate` | `Shard(1)` | OK | `Replicate` |

In-place `scatter_` does not raise on any of them, and the token shard is preserved
wherever the scores carry it. The stated mechanism does not exist.

The EP x TP failure this was written for was fixed later and for a different reason -- by
declaring the MoE an SP island with a replicated external boundary, which is what actually
gave the upstream layout something to redistribute to. The scatter branch has been dead
since.

## The one that stays, and why it is not the same case

`router_input_BLD` is a keyword-only argument defaulting to `None`; when it is `None` the
router reads `x_BLD` and every other model is bit-identical. K3's latent MoE needs the two
to differ because report Eq. 11 routes on the full-width token while dispatching a narrower
projection to the experts.

Worth noting what the upstream K3 PR did with the same requirement: it wrote
`KimiLatentMoE(Module)` -- its own router, its own dispatch -- and never touched
`common/moe.py`. That is the more conservative choice and it is the one migration step 3
adopts, at which point this argument comes out. Until then it is three lines that no other
model can observe.

## What was actually spent to learn this

Both premises were stated in a comment, in the tree, with a mechanism attached, for weeks.
Neither had been executed. The scatter one took four minutes to refute with a
twenty-line gloo script; the PP one took one reading of `trainer.py`. The general form,
already the first entry in `HOW_I_GET_THIS_WRONG_2026-08-13.md`: **a mechanism written
down in detail is not a mechanism that was checked**, and detail is not evidence.
