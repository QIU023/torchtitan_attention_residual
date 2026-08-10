# Finding 60 now has evidence, and both earlier records of it were wrong

Finding 60: the DEP vision stage exchanged the per-rank vision-sentinel counts and then
discarded them, never calling `_select_cp_shard` the way the non-DEP path does, so the
stage spliced every rank's visual features into its own sequence shard. The call was added
on 2026-08-09, and two records then disagreed about whether it mattered:

* the code comment claimed the missing call produced a step-1 forward 0.042 away from the
  non-DEP path;
* `OVERNIGHT_REVIEW_2026-08-09.md` recorded that the numbers never moved on any
  configuration tried, so "the fix is not supported by evidence".

Neither is right. The fix matters, and what decides whether it is visible is the sentinel
partition -- not depth, not the dynamic-CP mode, not the flavor.

## The control

`matrix_scripts/dep_no_cp_select_control.py` restores the pre-fix behaviour on the DEP path
only: overriding `_select_cp_shard` on `KimiK3ViTStage` leaves `KimiK3MultimodalModel`'s
own call intact, so a DEP-off arm still selects and stays a valid reference. It prints
`CONTROL_ENGAGED` with the partition it saw, because a control that silently never fires
reads exactly like "the fix changes nothing".

Shared seed checkpoint, `report_arch`, cp2 x pp2, 2 steps:

| seq | partition | DEP off | DEP on | DEP on, selection removed |
|---|---|---|---|---|
| 256 | `counts=[64, 0]` | 11.83717 / 12.1582 | identical | **identical** |
| 96 | `counts=[47, 1]` | 11.86731 / 17.2036 | identical | **ValueError** |

At seq 96:

    ValueError: 47 vision sentinel(s) in input_ids match neither the image count
    (1, one sentinel per image) nor the visual-token count (48, one sentinel per token)

Rank 0 holds 47 sentinels but receives all 48 features, and the convention detector
rejects the pair. So on a non-degenerate partition the pre-fix path does not merely drift
-- it cannot run.

## Why every earlier configuration said "nothing moved"

`counts=[64, 0]` is degenerate. Rank 0's slice IS the whole encode, so selecting or not
selecting gives it the same tensor; rank 1 holds no sentinels, so the features it wrongly
receives have nowhere to be spliced and are dropped either way. Both arms agree for a
reason that has nothing to do with the fix being unnecessary.

That is why the earlier attempt reached "no evidence" and stopped: the configurations it
had were all `[N, 0]`. Reaching a split partition needs the sentinels to straddle a shard
boundary, and here that took shortening the sequence below the visual-token count so the
48 tokens cross the midpoint of a 96-token sequence.

## The 0.042 does not reproduce

I could not produce a silent numerical divergence at all. On the split partition the
convention detector raises before any splice happens, and on the degenerate one the two
arms agree exactly. The 0.042 is withdrawn from the code comment rather than kept as an
unverified number. It may have been measured against the other splice convention
(`_splice`, one sentinel per image, expanded in place) which this collator does not use --
but that is a guess, and it is not recorded either way.

## Unit coverage, which is what was actually missing

`torchtitan/models/kimi_k3/tests/test_cp_shard_selection.py`, 6 CPU tests on
`_select_cp_shard` with no process group (the method reads `self._cp_group` only to ask
its rank, so a stub plus a patched `get_rank` covers it): an uneven split reconstructing
the encode exactly once in order, a rank holding no sentinels getting an empty slice, a
list of per-image features concatenated in order, a middle rank of three starting after
the lower ranks, a partition disagreeing with the encode raising, and `counts=None`
passing everything through.

## Repro

    S=/workspace/f60b
    base="--module kimi_k3 --config kimi_k3_debugmodel_report_arch --debug.seed 42 \
      --debug.deterministic --training.seq_len 96 --training.global-batch-size 4 \
      --training.local-batch-size 2 --parallelism.data_parallel_shard_degree 1 \
      --parallelism.context_parallel_degree 2 --parallelism.pipeline_parallel_degree 2"
    # seed checkpoint with -m torchtitan.train, then three arms from it:
    #   KIMI_VIT_DYNAMIC_CP=0                    -m torchtitan.train        (reference)
    #   KIMI_VIT_DYNAMIC_CP=0 KIMI_VIT_DEP=1     -m torchtitan.train        (fixed)
    #   KIMI_VIT_DYNAMIC_CP=0 KIMI_VIT_DEP=1     matrix_scripts/dep_no_cp_select_control.py
    # --parallelism.context_parallel_load_balancer takes None, not the string "none";
    # kimi_linear's CP path rejects a permuting balancer and says so.
