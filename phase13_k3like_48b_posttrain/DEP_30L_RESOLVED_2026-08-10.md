# The 30-layer DEP divergence was a cold-init comparison

`OVERNIGHT_REVIEW_2026-08-09.md` parked a 6.03e-04 step-1 divergence between DEP off and
DEP on on the 30-layer flavor, against 0.000e+00 on the 13-layer one, and recorded that
the chunked-loss control could not run because of an FSDP assertion. Both parts are now
settled: the divergence is in the initialization, not the computation, and the assertion
does not reproduce.

## The result

Same seed checkpoint, same flavor, same topology, 3 steps, loss and grad_norm:

| flavor / topology | arms | outcome |
|---|---|---|
| 30L `report_arch_pp8vp4`, seq 256, pp2 x dp2 x ep2, chunked(8) | DEP off vs on | **identical** |
| the same, plain CE instead of chunked | DEP off vs on | **identical** |
| 30L, seq 256, pp4 x dp2 x ep2, chunked(8) | DEP off vs `n_vit=1` vs `n_vit=2` | **identical** |
| 30L `bubble_ratio`, seq 4096, pp2 x dp2 x ep2, chunked(8) | DEP off vs on | **identical** |

The pp4 row is the stronger one: it includes the split tower, so clause 2 of the report's
DEP description is now verified at 30 layers and not only at 13.

Cold-initialized, with nothing else changed, the same arms differ:

| flavor | cold DEP off | cold DEP on | relative |
|---|---|---|---|
| 30L `report_arch_pp8vp4`, seq 256 | 12.02994 / 20.2093 | 12.03447 / 20.3780 | 3.8e-04 |
| 30L `bubble_ratio`, seq 4096 | 12.04270 / 20.8448 | 12.03069 / 20.2639 | 1.0e-03 |

Those are the recorded numbers' scale (6.03e-04, 8.3e-03), so the parked measurement was
a cold comparison.

## Why cold arms differ, shown directly rather than inferred

`init_digest_probe.py` prints every parameter's norm summed per rank, straight out of
`Trainer.__init__` and before any checkpoint load. On the pp2 x dp2 x ep2 topology:

    DEP off   rank0/1: 456 params, 4.084385566915e+03   rank2/3: 424 params, 3.647982193682e+03
    DEP on    rank0/1:  31 params, 5.360339825914e+02   rank2/3: 849 params, 7.197118015286e+03

880 parameters either way -- the same model -- but placed differently, and the totals are
7732.3678 against 7733.1520, a 1.0e-04 difference in the weights themselves before a
single forward runs.

That is expected rather than broken. DEP's whole mechanism is putting the vision tower on
a different pipeline stage, and torchtitan seeds PP ranks distinctly
(`distinct_seed_mesh_dims=["pp"]`), so moving a module to another stage hands it a
different seed. A cold DEP-on run is a different but equally valid initialization.

**What this means for the claim.** "DEP is numerically exact" is only ever testable from a
shared checkpoint; a cold A/B cannot test it, on any flavor. The 13-layer result was
already run that way, which is why it read 0.000e+00 while the 30-layer cold runs did not.
Depth was never the variable.

## The FSDP assertion does not reproduce

Recorded as the blocker:

    AssertionError: FSDP reduce-scatter expects uniform gradient dtype
                    but got {torch.float32, torch.bfloat16}

Five configurations run with plain CE, all clean: 30L pp2 x dp2 x ep2; 30L dp4 with no PP,
so reduce-scatter covers every parameter; 30L pp4 x dp2 x ep2 with the tower split across
two stages; and both chunked counterparts. I cannot say what the original attempt hit.

One nearby failure is real and worth separating from it: `bubble_ratio` unchunked at seq
4096 dies with `OutOfMemoryError: tried to allocate 2.50 GiB`, which is exactly the
condition chunked loss exists to avoid on that flavor (the fp32 logits upcast is
5.37 GiB at seq 8192). So on that one flavor the unchunked control genuinely cannot run
here -- for a memory reason, not a dtype one.

## Chunked and plain CE are not identical, and that is not a DEP question

Warm, DEP off, 30L pp2, 3 steps: chunked reaches 11.10841 / gn 11.8210, plain CE
11.10697 / gn 11.2963 -- 1.3e-04 on the loss. Chunked loss accumulates the hidden-state
gradient in float32 over 8 chunks, so the summation order differs. It shows up equally
with DEP off and on, which is what makes it a loss-implementation difference.

## Tooling added

* `matrix_scripts/unchunked_loss_control.py` -- `ChunkedLossWrapper` is pinned inside the
  flavor and torchtitan has no CLI override for `loss`, so "same flavor, unchunked" was
  not expressible on the command line. This patches the registry function in-process and
  swaps the wrapper for its own inner loss config, which also drops `_skip_lm_head`
  (the trainer only sets it for a chunked loss) and keeps `global_vocab_size`.
* `matrix_scripts/init_digest_probe.py` -- per-rank parameter digest out of
  `Trainer.__init__`, for separating "computes differently" from "starts differently".

## Repro

    S=/workspace/dep30
    # seed checkpoint, then DEP off/on from it
    torchrun --nproc_per_node=4 -m torchtitan.train --module kimi_k3 \
      --config kimi_k3_debugmodel_report_arch_pp8vp4 --debug.seed 42 \
      --debug.deterministic --training.seq_len 256 --training.steps 1 \
      --training.global-batch-size 8 --training.local-batch-size 2 \
      --parallelism.expert_parallel_degree 2 \
      --parallelism.data_parallel_shard_degree 2 \
      --parallelism.pipeline_parallel_degree 2 --checkpoint.enable \
      --checkpoint.interval 1 --dump-folder $S/seed
    # each arm adds --checkpoint.initial-load-path $S/seed/checkpoint/step-1
    #   --checkpoint.initial-load-model-only, and DEP on prepends KIMI_VIT_DEP=1
    # unchunked: run matrix_scripts/unchunked_loss_control.py instead of -m torchtitan.train
    # pp4 split tower: --parallelism.pipeline_parallel_degree 4, 8 ranks,
    #   global-batch 16 / local-batch 4, KIMI_VIT_DEP_STAGES=2
