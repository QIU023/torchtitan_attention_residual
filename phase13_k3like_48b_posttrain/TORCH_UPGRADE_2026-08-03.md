# torch dev20260725 -> dev20260802: PP restored, matrix bit-identical

## Why

torchtitan #3856 calls `pp_schedule.step(arg_mbs=..., kwarg_mbs=...,
target_mbs=...)`. The installed PyTorch's `PipelineScheduleSingle.step` had

    def step(self, *args, target=None, losses=None, loss_kwargs=None, **kwargs)

with no such parameters, so the list landed in `**kwargs` and `step` re-split it
via its own `_split_inputs` -- four microbatches became one. #3856 landed
2026-07-31; the box was on the 07-25 nightly.

After the upgrade:

    step params: self, args, target, losses, return_outputs, loss_kwargs,
                 arg_mbs, kwarg_mbs, target_mbs, kwargs

## Verification: all 13 legs bit-identical

    dp1                  7.72376 7.14969 6.22981
    fsdp2                7.67856 7.26904 6.29599
    pp2                  7.66117 7.04581 6.12275
    cp2                  7.69600 7.21146 6.26816
    tp2                  7.70027 7.04398 6.15859
    fsdp2_tp2_pp2        7.71036 7.23313 6.28071
    fsdp2_tp2_cp2        7.76396 7.31618 6.39315
    fsdp2_pp2_cp2        7.70254 7.23634 6.33279
    tp2_pp2_cp2          7.71961 7.15782 6.29697
    ep2_fsdp2            7.67856 7.26963 6.29357
    ep2_fsdp2_tp2_pp2    7.73785 7.21456 6.31275
    ep2_fsdp2_tp2_cp2    7.71987 7.22920 6.35607
    ep2_fsdp2_pp2_cp2    7.75924 7.24905 6.52276

Every value matches the pre-upgrade baseline exactly. K3's `fsdp2_pp2` gives
7.70898, identical to the pre-merge measurement.

Only `torch` was upgraded (`--no-deps`); torchvision has no dev20260802 build and
the pinned 0.29.0.dev20260726 still works.

## A third false alarm from the collector

The first post-upgrade matrix run reported all 13 legs FAILED. Nothing was wrong
-- the same commands by hand printed correct losses. The collector was not
stripping ANSI colour codes, so its `grep -oE "step: +[0-9]+ +loss: ..."` matched
nothing.

That is the third time tonight a collector manufactured a failure: PP's negative
placeholders being deduplicated ahead of the real value, `grep -v ' -'` removing
the surviving row, and now unstripped colour codes. Each one looked exactly like
a real regression. The rule that would have caught all three: before believing a
matrix result, run one leg by hand and compare.
