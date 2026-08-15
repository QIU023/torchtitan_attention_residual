# Porting TP/EP onto the upstream K3 tree -- measured, not estimated

Written after our tree's TP/EP declarative migration reached 54/54 with every
declarable module declared. Everything below is a count taken from the two trees,
not a guess. Our tree is `torchtitan/models/kimi_k3/`; the vendored upstream tree
is `torchtitan/models/kimi_k3_up/` (3067 lines).

## What upstream's tree already gives us

Their `parallelize.py` is 140 lines and raises `NotImplementedError` for both
tensor parallel and expert parallel. So TP/EP is entirely unwritten there. But
the two preconditions that would have been expensive are already satisfied:

* **Every class subclasses torchtitan's `Module`** (or `FeedForward`,
  `BaseAttention`, `GroupedExperts`, `Decoder`) -- 14 classes across `model.py`
  and `vision_encoder.py`. Declarations can attach without touching class
  hierarchy.
* **No Linear is constructed inline.** Every one is a `Linear.Config` field on
  the parent's Config, populated through a `_linear()` helper at **34 call sites
  in `__init__.py`**. `Linear.Config` already has a `sharding_config` field, so
  the declarations go in one file at 34 places.

That is a better authoring surface than ours, where the declarations sit at 39+13
constructor sites spread across `model.py` and `attn_res_model.py`.

## Where their tree is cheaper than ours

Their `KimiRMSNormGated` (model.py:50) is their own `Module` in plain PyTorch,
not fla's `FusedRMSNormGated`. On our tree those 9 `o_norm` modules are
permanently undeclarable, because a third-party class cannot hold a
`ShardingConfig` and its forward has to be shimmed to unwrap. On their tree the
same 9 modules are declarable, and the fla shim plus the `torch.compiler.disable`
around that op both become unnecessary.

So of our four undeclarable groups, theirs loses one:

| group | ours | theirs |
| --- | --- | --- |
| `router.gate` (12) | core's moe.py, not ours | same |
| `ffn.latent.norm` (12) | MoE output side, value arrives plain | same shape, needs re-measuring against their MoE contract |
| `o_norm` (9) | fla class, cannot declare | **declarable** |
| `embed_tokens` (1) | upstream's own vocab-parallel forward | same |

## Where their tree is more expensive -- and this is the real cost

Count of lines touching `DTensor` / `from_local` / `to_local` / `redistribute`
in the model files:

| file | ours | theirs |
| --- | --- | --- |
| model.py | 83 | 7 |
| vision encoder | 40 (multimodal_model.py) | **0** |
| attn_res_model.py | 22 | n/a (carrier is threaded in their block) |
| attn_res.py | 6 | n/a |
| total | **151** | **7** |

Their model code is written for plain tensors under FSDP2 only. The vision
encoder has not a single DTensor line. Every boundary we found has to be
re-created against their class layout:

* the SDPA output re-wrap (`from_local` on `Shard(2)`, then redistribute)
* the Ulysses CP exit re-wrap before `o_proj`
* three `_PlainGradBoundary` seals in the vision tower, including the one on the
  `all_gather` *output* that six earlier attempts missed
* `_splice`'s positional index write, because `masked_scatter` has no DTensor rule
* the fla shims for `ShortConvolution` and the KDA kernels

None of these were found by reading code. Each came out of a matrix run, and the
vision tower's took six attempts before `--debug.detect-anomaly` plus the insight
about the `all_gather` output resolved it. Re-creating a *known* boundary on a
different layout is much cheaper than finding it, but it is not free, because the
placement that is correct depends on what the surrounding module does.

## The imperative remainder, sized

Our five TP/EP functions in `parallelize.py`, code lines excluding comments and
docstrings:

| function | code | prose |
| --- | --- | --- |
| `apply_tp_kimi_k3` | 168 | 292 |
| `_apply_tp_moonvit_mlp` | 51 | 32 |
| `_sweep_remaining_to_replicate` | 28 | 43 |
| `_drive_declarative_sharding` | 22 | 23 |
| `apply_ep_kimi_k3` | 16 | 15 |
| total | **285** | 405 |

EP is 16 lines of code -- it delegates to core. The sweep is 28 and is not
optional: `clip_grad_norm_` needs every parameter on a consistent mesh, so
whatever the declarations do not claim has to become `DTensor(Replicate)`.

## So the work is

1. **34 `sharding_config=` arguments** in their `__init__.py`. Mechanical, but the
   names do not map 1:1 -- their `wq_a`/`wq_b`/`wkv_a`/`wkv_b`/`wo` are our
   `q_a_proj`/`q_b_proj`/`kv_a_proj`/`kv_b_proj`/`o_proj`, and their
   `forget_a`/`forget_b`/`beta`/`output_gate`/`output_proj` are our
   `f_a_proj`/`f_b_proj`/`b_proj`/`g_proj`/`o_proj`. Their MoE nests as
   `routed_down`/`routed_up`. The placements themselves transfer, since they are
   a property of the tensor's role, not of its name.
2. **~285 lines of imperative TP/EP** re-authored against their module names,
   into a `parallelize.py` that currently refuses TP and EP outright.
3. **~150 lines of DTensor boundary work in their model files**, where there are
   currently 7. This is the item that will consume the gate cycles.
4. **9 fewer exceptions** than ours to work around, thanks to their own
   `KimiRMSNormGated`.

Item 3 is the schedule. Items 1, 2 and 4 are a day's authoring; item 3 is
measured in gate runs, and the gate is 40 minutes for all 54 cells. Budget one
gate per subsystem boundary set -- text, vision, carrier -- plus the failures
that will not be predictable in advance, since none of ours were.
