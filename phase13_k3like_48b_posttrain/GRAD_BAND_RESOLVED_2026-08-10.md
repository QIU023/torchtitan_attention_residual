# The gradient band was the instrument: PP is bit-exact

Open since 2026-08-02, under three successive explanations -- "LoRA is not usable on
PP/CP", then "an AttnRes question", then (mine, earlier today) "it is KDA". All three
wrong. The band is an artifact of comparing two runs that were fed different per-forward
batch shapes.

## The result

Same warm checkpoint, same seed, same flavor, 6 steps, the only change being that the
reference and the PP leg see the same per-forward batch:

| flavor | instrument as written | shapes matched |
|---|---|---|
| `mini_block_attn_res` (21L, KDA, MoE, AttnRes) | `max>1%` **0.16749** | **0.000000** over all 592 parameters |
| `diag_4l_kda` (4L, all KDA, AttnRes) | 0.00272 | **0.000000** over all 97 parameters |

Not "small". Exactly zero, every parameter.

## What the instrument was doing

`lora_vs_fullparam_axes.sh` and its descendants run:

    leg ref 1 2   --parallelism.data_parallel_shard_degree 1
    leg pp2 2 4   ... --parallelism.pipeline_parallel_degree 2

The third field is `local_batch_size`. So the reference does one forward at batch 2 while
the PP leg runs microbatches of 1 -- a different tensor shape into every kernel. The legs
have to differ in batch size for PP to have enough microbatches, which is why it was set
that way, and the shape consequence was not noticed.

Matched arms are `ref` at `local_batch_size 1` against `pp2` at `local_batch_size 4` with
microbatch size 1: eight accumulation steps of 1 against two accumulation steps of four
microbatches of 1. Same shape into every kernel, and the gradients agree exactly.

## Why only KDA showed it

`chunk_kda` is a chunked scan: its blocking depends on the input shape, so a different
batch shape changes the summation order inside the kernel. Dense matmuls do not care.
That explains the whole magnitude pattern, which is why it was so easy to read as
causation:

| arm | pp2 `max>1%` | reading |
|---|---|---|
| 4L MLA, no AttnRes | 0.00015 | shape-insensitive |
| 4L MLA, AttnRes | 0.00056 | AttnRes mixes blocks, so it concentrates whatever is upstream |
| 4L all-KDA, no AttnRes | 0.00105 | the scan is shape-sensitive |
| 4L all-KDA, AttnRes | 0.00272 | both |
| 21L MLA, AttnRes | 0.00079 | depth alone does nothing |
| 21L, 15 KDA, MoE, AttnRes | 0.16749 | 15 shape-sensitive layers, compounded |

Every one of those numbers is real and reproducible. They are all zero once the shapes
match, so they measure the kernel's response to a shape change, not the parallelism.

## Three wrong attributions, and what each one got wrong

* **LoRA** (2026-08-02, withdrawn 2026-08-08). Compared across flavors, so the arms
  differed in more than LoRA. What LoRA actually changes is composition: the AttnRes
  pseudo-queries hold 10.1% of the gradient norm full-parameter and 76.4% under LoRA,
  because the frozen base contributes none, so the same ill-conditioned parameters clear
  the ">1% of norm" filter that had been hiding them.
* **AttnRes** (the standing explanation this morning). The worst-parameter column names
  `attn_res_proj`, which is where the contamination shows rather than where it starts --
  the same parameter sits at 0.0025 with KDA removed, and at 0.000000 with shapes matched.
* **KDA** (mine, an hour ago). A 67x drop when KDA is removed is a real measurement and I
  read it as cause. It is the kernel's shape sensitivity being exposed by a confound in
  the arms; remove the confound and KDA is exact too.

The common shape of all three: a single-variable arm was run, the number moved, and the
variable was named as the cause -- without checking whether the two arms were equivalent
in everything else. The shape difference was in plain sight in the runner's arguments the
whole time.

## What is now established, and what is not

**Established.** For this model, PP produces bit-identical gradients to a single-rank
reference when the per-forward shape is held constant -- 592 parameters, 21 layers, KDA and
MoE and AttnRes all present. That is a much stronger statement about the PP implementation
than the 18-cell loss matrix makes, because the matrix compares losses within a cell across
code changes and never compares one parallelism against another.

**Not established, and not reachable the same way.** CP cannot be shape-matched: sharding
the sequence is what CP does, so each rank necessarily sees `seq/cp` and the scan blocks
differently. Its recorded deviation (0.16260 on the same flavor) needs a different control
-- comparing cp2 against cp2 with a different rank assignment, or against an fp32 reference
where the scan's summation order matters less. TP is in the same position via head
sharding.

**Retired.** "PP and CP are not usable under LoRA without further work" -- withdrawn in
stages, and now with a positive result for PP rather than only the absence of evidence.

## Repro

    # 21-layer arm, shapes matched
    SP=$(find /workspace/attnres_band/mini_bf16_warm -type d -name step-3 | head -1)
    base="--module kimi_k3 --config kimi_k3_mini_block_attn_res --training.seq_len 512 \
      --debug.seed 42 --debug.deterministic --training.dtype bfloat16 \
      --checkpoint.enable --checkpoint.initial-load-path $SP \
      --checkpoint.initial-load-model-only --training.steps 6 \
      --training.global-batch-size 8"
    # ref:  --training.local-batch-size 1, 1 GPU
    # pp2:  --training.local-batch-size 4, 2 GPUs, --parallelism.pipeline_parallel_degree 2
    # both through tp_trainer_grad_probe.py with GRADCHK_DUMP set, then grad_band_compare.py
