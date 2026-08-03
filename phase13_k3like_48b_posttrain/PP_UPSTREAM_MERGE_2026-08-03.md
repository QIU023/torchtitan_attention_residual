# The upstream merge breaks PP, and it is the microbatch contract

## Gate result

The PP legs asked for after the merge both fail, on BOTH branches:

    fsdp2_pp2   ValueError: Expecting 4 arg_mbs but got 1
    pp2_cp2     ValueError: Expecting 4 arg_mbs but got 1

Confirmed identical on `attention_residual_dev` and on `k3_models_move`, so this
is the upstream merge and not the experiments/ -> models/ move.

The 07-30 baseline for `fsdp2_pp2` is 7.71304; it cannot currently be
reproduced.

## What changed upstream

`pp_forward_backward_step` now takes microbatch LISTS -- `input_dict_mbs`,
`label_mbs` -- and builds `arg_mbs` / `kwarg_mbs` / `target_mbs` from them, one
entry per gradient-accumulation step:

    for input_dict, labels in zip(input_dict_mbs, label_mbs, strict=True):
        ...
        arg_mbs.append((inputs,))

and `forward_backward_step` asserts `isinstance(input_dict, list)` under PP. So
the number of microbatches the schedule sees is now the number of accumulation
steps, where it used to come from splitting the local batch.

## Why parameters cannot fix it

The two constraints pull against each other:

- the schedule wants `n_microbatches >= n_stages`, and it derives that count
  from the accumulation loop;
- `Number of microbatches (1) must be greater than or equal to ...` appears when
  the local batch is shrunk to raise the accumulation count.

Sweeping global/local batch only changes the number in the error -- "Expecting 4"
becomes "Expecting 2" -- because the accumulation loop still yields a single
dict per step on this path. The K3 train path is not producing the list the new
contract expects.

## It is upstream's own PP, not ours

`deepseek_v3_debugmodel` fails identically on the merged tree:

    NGPU=2, pp2, local_batch 4, global 8
    ValueError: Expecting 4 arg_mbs but got 1

So the break is not in the K3 path and not in our adapter. Adding
`--parallelism.pipeline_parallel_microbatch_size 1` does not help either model.

## Traced to the schedule, and it is not ours

Instrumenting the trainer settles where the count is lost. Rank 0, pp2,
local_batch 4:

    [MB4] input_dict_mbs=4 label_mbs=4          <- the list arrives complete
    [MB5] loop iter -> arg_mbs=1 ... first_stage=True
    [MB5] loop iter -> arg_mbs=2
    [MB5] loop iter -> arg_mbs=3
    [MB5] loop iter -> arg_mbs=4                <- and is built complete
    ValueError: Expecting 4 arg_mbs but got 1   <- yet the schedule sees one

So the trainer does everything right and `pp_schedule.step` reports one. The
error text comes from PyTorch's schedule, not from torchtitan.

It is also not the K3 adapter, which wraps `pp_schedule.step`
(pipeline_adapter.py:983). Two greps: `deepseek_v3` references nothing from that
module, and nothing outside `models/kimi_k3` imports `pipeline_adapter` at all.
deepseek_v3 fails identically without ever touching it.

That leaves a contract mismatch between torchtitan's new
`pp_forward_backward_step` and the installed PyTorch schedule -- plausibly a
version skew, since this box runs torch 2.14.0.dev20260725+cu130 rather than the
2.12 line this merge was developed against. Not verified, and stated as the
remaining candidate rather than a conclusion.

## Where the mismatch is

The trainer computes the count correctly:

    num_pipeline_parallel_microbatches = local_batch_size // pp_microbatch_size
                                       = 4 // 1 = 4

and the accumulation loop does fetch that many:

    for _ in range(self.gradient_accumulation_steps):
        microbatches = []
        for _ in range(self.num_pipeline_parallel_microbatches):
            input_dict, labels = next(data_iterator)
            microbatches.append((input_dict, labels))

Its init line confirms the arithmetic -- "local batch size 4, global batch size
8, gradient accumulation steps 2". Yet the schedule receives one entry. So each
`next(data_iterator)` is yielding a whole local batch where the new contract
wants one microbatch: the dataloader side of this change has not landed for the
`c4_test` path, or needs a knob neither model sets.

Reported rather than patched. The fix belongs wherever upstream intended the
split to happen, and guessing at it from the consumer side would put the K3
adapter -- the component with the clean per-parameter record -- on the hook for
an upstream contract change.

## Not attempted

Patching the adapter or the loop to synthesize a list would be guessing at a
contract that upstream has just rewritten, on the one component with a clean
per-parameter record (PP adapter: 0.00000 over 548 parameters across pp2, pp4,
pp2xvp2, pp4xvp2). The right move is to read the new contract end to end against
a working upstream PP run first.

Everything else survives the merge: the conflict was one line in
`_supported_models` (upstream added `kimi_k2_7`, we added `kimi_k3` -- both
kept), no script imports the removed `set_step_loss_kwargs`, and the non-PP axes
are unaffected.


## Bisected: upstream #3856 "Always Pre-Split Microbatches for PP"

`git bisect` over the 34 commits the merge brought in, probing with
`deepseek_v3_debugmodel` at pp2, lands on `9228564523aa`:

    Always Pre-Split Microbatches for PP (#3856)
    - Have PP training and validation dataloaders emit pipeline microbatches directly.
    - Keep PP microbatching distinct from gradient accumulation and pass pre-split
      args, kwargs, and targets through the upstream schedule API.

The title is the diagnosis. It changed torchtitan/trainer.py (184 lines),
components/validate.py, config/configs.py, and every model or experiment that
carries its own trainer -- flux, torchft, forge, graph_trainer.

### A false bisect first, and why

The first run reported `ab461244d "Make AutoParallel tests GPU-agnostic"`, which
touches two files under experiments/graph_trainer and cannot affect PP. Every
step had been marked bad, including commits that could not possibly break it.

The probe was the problem: it ran `--module kimi_k3`, and across the bisect range
kimi_k3 sits in `experiments/` and is registered in `_supported_experiments`,
while the probe's resolution path expects `_supported_models`. So every commit
failed on module resolution, never reaching PP. Re-probing with `deepseek_v3` --
present and unmoved across the whole range -- converged on #3856 in three steps.

Worth keeping as a rule: a bisect probe has to be valid at BOTH ends before the
run, and "good end passes" is the check that catches this. It was not run the
first time.

### What this does not yet explain

K3 has no trainer of its own -- no `trainer.py` or dataloader under
`models/kimi_k3` -- so it uses the core trainer that #3856 already updated. The
instrumentation above shows that core trainer building `arg_mbs` correctly to 4
and the schedule still seeing 1. So "our path did not follow the change" is not
the explanation, and neither is the earlier version-skew guess: PP was verified
per-parameter at 0.00000 on this same torch nightly BEFORE the merge.

The remaining question is narrow and answerable: what does #3856 expect the
dataloader to emit, and does the c4_test path K3 and deepseek_v3 share emit it?
