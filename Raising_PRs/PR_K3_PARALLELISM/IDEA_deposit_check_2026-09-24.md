# Idea (no change yet): the deposit count check, and what it should become

Raised by shuhuayu on PR 4312 (r4080717964, 2026-09-23, non-blocking): "this is expecting every stage must use the
cached blocks on it, so it is required to contribute a grad. so there is a splitting constraint that each stage must
have a transformer layer, which is satisfied by the titan stage splitting, but not true in general for any stage
splitting." The user promised a comment in the next revision; round 5 adds it (commit "kimi_k3: the deposit count
check names the split it assumes"). This note is the longer-term view, recorded only; nothing below is on a branch.

## What the check does today

- `BlockLayoutTables` calls a stage a reader of block b when b is in its rank's store as the stage starts
  (`cache_at_entry`); `deposits_expected(b, owner)` counts the readers on the owner's rank after the owner.
- In backward each stage splits its stack gradient (`_split_stack_grad`): received columns go back on the wire,
  held columns are deposited into `PPRankLocalCache`; the owner collects them in `_collect_into` and raises if the
  count differs from the tables.
- The count is the only thing that turns a protocol slip (a deposit landing after its owner collected, a stage whose
  backward never ran) into an error instead of a silently wrong gradient, and it needs nothing from the schedule but
  the per-rank stage order. It should stay loud.

## The exact condition (measured on the CPU harness, 2026-09-24)

A stage deposits only if its stack leaf gets a gradient, i.e. the stack reaches its backward through a transformer
layer, the output aggregation, or a non-empty outgoing delta (the sent columns make the leaf's gradient dense, with
zeros in the held columns nobody read). So the constraint is narrower than "every stage has a layer":

- `test_kimi_k3_pp_block_grads.py` with a 12-stage pp4 split, local copies only:
  - P: an empty stage 4 whose outgoing delta is [1], plus the head alone on the last stage: passes, bitwise against
    the single-device closed form.
  - F: empty stages 9 and 10 whose outgoing deltas are []: fails with "stage 6 micro-batch 0 block 3: 0 gradient
    deposit(s) but 1 expected; a later stage on this rank did not run its backward".
- The message is wrong about the cause in F: stage 10 ran its backward; its stack got no gradient.
- No shipped split hits it: core's `_generate_llm_fqn_per_model_part` gives every middle stage a layer and the last
  one the head (Kimi K3 pins the aggregation there), and DEP's tower-only stage is stage 0, which holds no cache.

## Options

1. Validate the split at build time in `pipeline_kimi_k3`: every stage from index pp on needs a layer or the
   aggregation. Early and explicit, but it keeps the constraint and is stricter than needed (P would be refused).
2. Deposit zeros when the stack has no gradient. Removes the constraint and keeps the counts, at a `[T, D]` zero per
   held block per micro-batch on such stages.
3. Count reports, not tensors (preferred). `_split_stack_grad` returns an entry for every held block, the gradient
   column or None when the stack has no gradient, and `PPRankLocalCache.deposit` counts a None without adding. The
   check stays exact (a missing or extra report still raises), the constraint disappears, nothing is allocated, and
   the tables stay free of model knowledge. About ten lines in `stage.py` and `cache.py`, split F as the regression
   case in `test_kimi_k3_pp_block_grads.py`, and the error message changed to name what is left: a later stage on
   this rank whose backward did not run.
4. Tell the tables which stages read the stack (from the split: a `layers.*` FQN or the last-stage modules) and count
   only those. Puts model knowledge into the pure routing tables, and still miscounts a non-reading stage that sends
   part of the stack on (P), which would deposit zeros nobody expects.

## When

After 4312 merges, in the first K3 PR that can place a non-layer stage after stage 0, which the user named to
shuhuayu as the K3 MoonViT PP DEP PR. Option 3 there, with F as its test; option 1 is then unnecessary.
