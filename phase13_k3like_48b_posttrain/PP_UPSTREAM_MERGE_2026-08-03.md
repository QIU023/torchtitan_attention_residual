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


## Traced into #3856: rank 1 sends an empty arg_mbs

Instrumenting `pp_schedule.step` on EVERY rank (the earlier probe only printed
rank 0, which is why this took two passes):

    [R0] first=True  last=False  arg_mbs=4  kwarg=4  target=None  n_mb=4
    [R1] first=False last=True   arg_mbs=0  kwarg=4  target=4     n_mb=4
    ValueError: Expecting 4 arg_mbs but got 1

Rank 0 is consistent -- 4 microbatches, schedule expects 4. Rank 1 is not a
first stage, so `forward_backward_step` skips `arg_mbs.append(...)` and passes
an EMPTY list rather than `None`:

    arg_mbs=arg_mbs if self.pp_has_first_stage else None

reads as if it passes None, but `arg_mbs` is `[]` on a non-first stage, and `[]`
is falsy only in a boolean test -- this is a conditional expression, so the
empty list is passed. PyTorch's `_check_inputs` then takes the
`arg_mbs is not None` branch and rejects the length.

`torchtitan/distributed/pipeline_parallel.py` is untouched by #3856, so the
schedule's own `n_microbatches` (4, verified) is right; the defect is on the
feeding side.

The "got 1" in the message does not match either rank's count (4 and 0), so
there is one more layer between these prints and the raise -- plausibly a
per-stage schedule in the multi-stage path. Not chased further tonight; what is
established is that the empty-vs-None distinction on non-first stages is real
and is upstream's, since deepseek_v3 shows it with no K3 code involved.


## Root cause: #3856 needs a newer PyTorch than this box has

The last layer is a signature mismatch, not a logic bug. Instrumenting
`_check_inputs` on every schedule class shows what actually arrives:

    [CI-PipelineScheduleSingle] rank=0 n_mb=4 arg=1 kwarg=1
    [CI-PipelineScheduleSingle] rank=1 n_mb=4 arg=1 kwarg=1

One, from both ranks, where the trainer passed four. The installed PyTorch has

    def step(self, *args, target=None, losses=None, loss_kwargs=None, **kwargs)

with no `arg_mbs` parameter at all. torchtitan #3856 calls

    self.pp_schedule.step(arg_mbs=..., kwarg_mbs=..., target_mbs=..., ...)

so `arg_mbs` lands in `**kwargs` as a single value, and `step` then runs its own
`_split_inputs(args, kwargs)` over it -- splitting a 4-element list into one
microbatch. Hence "got 1" matching neither rank's count: it is the length after
PyTorch re-split something that was already split.

Dates settle it. This box runs `torch 2.14.0.dev20260725`; #3856 landed
2026-07-31 and targets the newer schedule API. `2.14.0.dev20260802` is available
on the nightly index.

So there is nothing to fix in torchtitan or in our code -- the tree is correct
and the environment is six days stale. That also retracts the earlier framing of
this as "an upstream defect worth reporting": deepseek_v3 failing here is a
version skew on our side, not a bug upstream would accept.

Upgrading torch is not free on this box -- installing vLLM previously downgraded
torch and broke the training stack (DataParallelMeshDims disappeared; torchvision
had to be reinstalled). So the upgrade is worth doing deliberately, with the KDA
shared-memory behaviour and the fla kernels re-checked afterwards, rather than as
a side effect of chasing PP.
