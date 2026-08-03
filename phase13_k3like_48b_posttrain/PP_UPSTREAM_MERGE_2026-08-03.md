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
