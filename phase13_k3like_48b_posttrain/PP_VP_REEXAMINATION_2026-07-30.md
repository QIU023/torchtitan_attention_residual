# PP re-examined on all 8 GPUs, including VP != 1 (2026-07-30)

The earlier claim "PP is numerically transparent" rested on pp=2 with the default
VP=1, on 2-4 GPUs. Too narrow on two counts: this repo's own history cites
PP8 x VP4 validation, so VP is load-bearing, and a 2-stage split exercises one
boundary where pp8 exercises seven. Re-run with all 8 GPUs busy on every leg,
every leg loading the same seed checkpoint.

## Results

| leg | global/local batch | step-1 loss | grad_norm | verdict |
| --- | --- | --- | --- | --- |
| `fsdp8` (reference) | 8 / - | 7.71481 | 8.4704 | PASS |
| `dp4_pp2` | 8 / 2 | 7.71541 | 8.7562 | PASS |
| `dp2_pp4` | 8 / 4 | **7.71304** | 8.4957 | PASS |
| `dp2_pp4_vp2` | 8 / 1 | **7.71304** | 8.4957 | PASS |
| `dp2_pp2_cp2_ep2` | 8 / 2 | 7.71135 | 8.5883 | PASS |
| `dp1_pp8` | 8 / 8 | -- | -- | FAIL (hardware) |
| `dp1_pp8_vp2` | 16 / 16 | -- | -- | FAIL (layer tiling) |

## The invariant: loss tracks the DP degree and nothing else

    dp8 -> 7.71481    dp4 -> 7.71541    dp2 -> 7.71304    dp2+cp2 -> 7.71135

`dp2_pp4` and `dp2_pp4_vp2` both land on 7.71304, the dp2 value, and `dp4_pp2`
lands on 7.71541, the dp4 value (matching `ep2_fsdp4` from the previous matrix).
So:

* **PP is numerically transparent at degree 2 AND degree 4** -- not just the
  2-stage case the earlier claim rested on;
* **VP = 2 is numerically transparent too** (`dp2_pp4_vp2` == `dp2_pp4`), which is
  new: VP was previously untested here despite being cited in this repo's history;
* PP composed with CP and EP simultaneously still lands on its dp group's value.

The earlier claim survives and is now much better supported, but it was
over-claimed at the time it was made.

## Two failures, two different causes, neither a PP logic error

**`dp1_pp8` -- hardware.** `Failed to set the allowed dynamic shared memory size
to 108160` against the 5060 Ti's 101,376 B. Same fla KDA ceiling already recorded.
Note this is the dp_shard=1 pattern: `pp2` alone failed the same way in the
previous matrix, while `pp2_cp2` passed because CP halves the local sequence and
the autotuner picks a config that fits. Every PP leg that fails on this box has
dp_shard=1; every PP leg with dp_shard >= 2 passes.

**`dp1_pp8_vp2` -- layer-count tiling, and it is not a k3mini quirk.**

    ValueError: Number of virtual stages (12) must be divisible by pipeline
    parallel size (8). Model has 21 layers with
    pipeline_parallel_layers_per_stage=2.

21 = 3 x 7 does not tile 8 ranks at any useful layers_per_stage. **This applies to
the real config too**: K3 has 93 layers, and 93 = 3 x 31, so
`ceil(93 / layers_per_stage)` is divisible by 8 for no small value either. Any
PP8 x VP plan for the 2.8T model has to either accept uneven virtual stages or set
explicit per-stage module lists (`module_fqns_per_model_part`) rather than a
uniform layers_per_stage. Worth knowing before scaling out, and it is a
config-planning constraint rather than a defect.

## Method note

Four legs in the first pass failed on `Number of microbatches (1) must be >= the
number of stages (N)` -- microbatches equals `local_batch_size`, so pp4 needs >= 4
and pp8 >= 8, and global_batch must cover dp x local. Those were harness config
errors of mine, not code failures, and they are corrected above. Third time in
this logbook that a leg reported as failing turned out to be the harness; the
rule stands that a failure must be reproduced in isolation before it is believed.
