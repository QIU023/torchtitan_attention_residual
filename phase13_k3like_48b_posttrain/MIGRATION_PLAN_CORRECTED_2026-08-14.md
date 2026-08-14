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
