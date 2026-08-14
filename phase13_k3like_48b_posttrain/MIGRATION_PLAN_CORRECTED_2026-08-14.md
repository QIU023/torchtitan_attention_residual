# The migration plan, corrected -- read this before the older step docs

Two things were wrong in how the migration was being run today, and both were caught by
the human rather than by me.

## 1. Every matrix result must name the tree it ran on

I reported "54/54" for the carrier migration. True, and measured on **our** tree
(`torchtitan/models/kimi_k3/`), which already passed 54/54 before any of this. It says
the upstream carrier shape did not break our tree. It says **nothing** about migration
progress.

The upstream tree (`torchtitan/models/kimi_k3_up/`) is at **0/54** and cannot reach 54
today:

| the upstream tree needs | state | cells it gates |
|---|---|---|
| FSDP | shipped | baseline |
| CP (Ulysses + KCP) | implemented, a few cells checked | 5 |
| PP (`pipelining_fn` + carrier) | wired, **never run successfully** | 8 |
| EP | **absent** | 6 |
| TP | **absent** | 8 |
| **LoRA** | **absent** | the whole `mm_lora` arm, 18 |

LoRA was previously filed as "part of step 3, can wait". Wrong: it is a **precondition**
for 54/54 on that tree, since one of the three arms is a LoRA flavor. Five workstreams,
not four.

`mm_full` needs no new flavor -- their `debugmodel` is our `report_arch` in every free
parameter.

## 2. Order: finish declarative in OUR tree, then swap modules

Measured rather than argued.

Remaining imperative surface in our tree:

* `parallelize_module`: **4 real call sites** (line 704 vision tower, 861, 882, 1120 --
  the other two greps are an import and a comment)
* `use_local_output`: 38
* modules already carrying a `sharding_config`: **29**

Two candidate orders:

**A -- port each parallelism onto their tree** (what today was doing). Every one of the
first four steps runs on a tree that **cannot express the three-arm matrix**, so each is
gated on "it trains" rather than on 54/54. Today's error rate was concentrated exactly
where a criterion was missing or invalid.

**B -- finish declarative in our tree, then swap module implementations.** Every step runs
where 54/54 is expressible. And once sharding is declared on the config tree, swapping a
module **carries its parallelism with it** instead of needing the parallelism re-ported.

**B is faster, and not because it is less work.** A's first four steps have no gate, and
an ungated step produces conclusions that get redone.

The carrier migration is the worked example of B: adopt their `[T, N, D]` threaded
carrier inside our model, keep our adapter and our parallelism, gate on 54/54 plus a
bitwise cache-on/off comparison. It landed in one pass.

What blocked B before today is now gone: the KIND probe located the residual-stream kind
mixing, and the carrier change removed it -- a Python list cannot be declared, a threaded
tensor can.

### Order

1. **Declarative in our tree**: the 4 `parallelize_module` sites and the 38
   `use_local_output` sites, each step gated on 54/54 (our tree).
2. **Swap modules to the upstream classes**, one at a time, each gated on 54/54:
   `KimiMoE` -> `KimiLatentMoE` + `KimiGroupedExperts` (also the only way to get
   `router_input_BLD` out of `common/moe.py`), then the attention classes, then the
   vision encoder.
3. When the swap is complete the migration is done: our tree IS their structure, with
   declarative parallelism that travels with the modules.

### What the work on their tree is still worth

Not wasted, just out of order. The Ulysses/KCP attachment patterns, the PP carrier's
bitwise verification, and the thin `pipelining_fn` all apply at step 2 -- when the modules
being wired are the ones that will actually ship.

---

## Step 1 started: which of the four imperative sites can move, and why

Branch `migrate_tp_attnres_tail`. Measured, not reasoned.

### The rule that decides it

A module can go declarative in isolation **iff its parameter is read at a use site that
already unwraps a DTensor**. If the module is CALLED (`mod(x)`), a declared weight meets
the plain residual stream inside the op and dies with `aten.mul.Tensor got mixed
torch.Tensor and DTensor` -- the declarative vocabulary has `state_shardings`, `in_src`,
`in_dst`, `out_src`, `out_dst` and `local_map`, and **none of them is `to_local` on the
output**. That is why `use_local_output=True` has no declarative equivalent.

Demonstrated on the AttnRes tail, whose two modules both already carried
`sharding_config=_tp_replicate()` (the imperative `NoParallel` plan was making the
declarative driver skip the subtree, so the declaration was inert):

| declared | result on `tp2` |
|---|---|
| `final_attn_res_proj` + `final_attn_res_norm` | **FAILS** -- mixed Tensor and DTensor |
| `final_attn_res_proj` only, norm left imperative | **trains**, step 1 identical to baseline, later steps 1e-5 |

`proj.weight` is read directly inside `block_attn_res`, which already calls `to_local()`
on it. `norm` is invoked as a module. Same rule, opposite outcomes.

### What this implies for the remaining sites

| site | nature | movable alone? |
|---|---|---|
| L704 vision tower | 92-line `_apply_tp_moonvit_mlp`, its own CP mechanism | out of scope for this migration |
| L861 model level | `embed_tokens` / `lm_head` | embeddings switch to VOCAB-PARALLEL execution once `parallelize()` sets `tp_group` -- a different mechanism, established earlier at the cost of 29 test failures |
| L882 AttnRes tail | two modules | **proj yes, norm no** (measured above) |
| L1120 LoRA TP | per-layer plan with `.base` keys | untested |

So the 38 `use_local_output` sites are not 38 independent edits. Each one is a module
whose output feeds the residual stream, and **the stream has to flip together** -- which
is the same conclusion the KIND probe reached from the other direction. The tensor carrier
made that flip expressible; it did not perform it.

### Honest state of step 1

One of four sites is half-migrated on a WIP branch, verified on `tp2` only, not on 54/54.
The 1e-5 drift on later steps wants explaining before this is called done: a declared
Replicate and `NoParallel(use_local_output=True)` should be the same placement, so the
difference is a routing detail and not obviously benign.
