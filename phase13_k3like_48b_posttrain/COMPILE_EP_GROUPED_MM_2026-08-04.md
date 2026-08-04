# torch.compile + EP: `_grouped_mm` rejects a zero-sized operand

Refs: pytorch/torchtitan#3029

## What publishes, and what does not

pytorch/torchtitan#4025 states `torch.compile` is out of scope ("No packed
documents, activation checkpointing, `torch.compile`, or CPU..."), and passes
`compile_config=CompileConfig()`, whose default is `enable=False`. Our matrices
are compile-off too, so the published comparison needs no adjustment.

This records the compiled run anyway, because "we only tested the configuration
we publish" is how defects survive.

## Compiled 13-leg matrix: 11 pass, 2 fail

`kimi_k3_mini_diag_4l_moe_depth`, 3 steps, seed 42, deterministic,
`--compile.enable`.

Where both run, compile changes nothing beyond arithmetic reordering:

| leg | max abs(eager - compiled) |
|---|---|
| fsdp2 | 0.01308 |
| ep2_fsdp2 | 0.01139 |
| cp2 | 0.01044 |
| fsdp2_tp2_pp2 | 0.00984 |
| fsdp2_tp2_cp2 | 0.00790 |
| fsdp2_pp2_cp2 | 0.00694 |
| ep2_fsdp2_tp2_cp2 | 0.00606 |
| tp2_pp2_cp2 | 0.00605 |
| dp1 | 0.00272 |
| pp2 | 0.00165 |
| tp2 | 0.00054 |

Worst 0.013, against a 0.10-0.40 spread across the parallelism configurations
themselves. So compile is numerically fine on these eleven.

Two fail, both EP with pipeline parallel:

    ep2_fsdp2_tp2_pp2   RuntimeError: Invalid strides/sizes, got [0, 1] for
                        strides and [224, 0] for sizes
    ep2_fsdp2_pp2_cp2   same

## Where it comes from

    File "/tmp/torchinductor_root/s4/cs4nary....py", line 63, in call
      buf3 = extern_kernels._grouped_mm(buf2, primals_8, cumsum, out_dtype=None)
    RuntimeError: Invalid strides/sizes, got [0, 1] for strides and
                  [224, 0] for sizes

One operand has a zero-length dimension. The call site is
`torchtitan/models/common/moe.py:95,101,106` -- three `torch._grouped_mm` calls
with no guard for an empty group:

```python
h_RF = F.silu(torch._grouped_mm(x_RD.bfloat16(),
                                w1_EFD.bfloat16().transpose(-2, -1),
                                offs=offsets_E))
```

Confirmed that the kernel is intolerant of degenerate shapes: calling
`torch._grouped_mm` directly on a `[224, 0]` operand fails in eager too (with a
different message, "strides should be multiple of 16 bytes").

## Why eager survives it

Not established. Eager passes both legs on the identical configuration, so
either eager never materializes the zero-length dimension or it reaches
`_grouped_mm` with different shapes. Compile is the difference, but *what* it
changes has not been traced.

Stating that rather than guessing: the natural story -- "a rank gets zero
routed tokens and dynamo specializes away a guard" -- is untested, and there is
no such guard in `moe.py` to specialize away.

## Ownership: ours, not upstream's

That call site is byte-identical in this fork and in #4025's tree, which made
"core limitation, not ours" the obvious reading. It is wrong.

The rest of the file is NOT identical: this fork carries two changes to
`models/common/moe.py`, and one of them is in the routing path that feeds these
offsets --

    064a6e378  kimi_k3: wire Stable LatentMoE end to end (report Eq. 11)
    129e29de0  moe: preserve router shard placement in routing-map scatter
               (fixes TP+EP)

The second rewrites the `routing_map_BLE` scatter under TP+EP, and that map
determines `num_tokens_per_expert_E` -> `offsets_E` -> the group boundaries
`_grouped_mm` is handed.

Control run, on #4025's tree (which carries upstream's unmodified `moe.py`),
same parallelism and same compile flag:

    deepseek_v3_debugmodel, dp2 x ep2 x tp2 x pp2, --compile.enable
    -> step 1 loss 8.13452, passes

So the same core call site, the same parallelism, and the same compile setting
work upstream. **The defect is in this fork.** Prime suspect is the routing-map
change, which is exactly where an expert can end up with a zero-token group.

Caveat on the control: `deepseek_v3_debugmodel` has different expert dimensions,
so it establishes that this configuration is not generically broken upstream --
not that an identical routing distribution was exercised.

## Not fixed

Localized to this fork and to the EP+compile combination; eager is unaffected
and nothing currently published depends on it. Next step is to instrument
`num_tokens_per_expert_E` under the failing configuration and find which expert
goes empty, rather than adding a guard that would hide the cause.

---

# Addendum: the CP+TP hang on the PR-4025 twin

Refs: pytorch/torchtitan#3029

Two collective desynchronizations were found and fixed (commit 524bffdde): the
sentinel-count `all_reduce` sat behind a data-dependent early return, and an
image-free batch skipped the now-FSDP-sharded tower and with it the all-gather
FSDP2 issues from its pre-forward hook. Neither was sufficient --
`fsdp2_tp2_cp2` and `ep2_fsdp2_tp2_cp2` still hang at step 2.

Per-rank instrumentation of the entry to `_exchange_sentinel_counts` settles
what is and is not wrong.

**The shard arithmetic is correct.** rank 0 and rank 2 are a CP pair and report
`local=255` and `local=34`. They sum to 289, which is exactly 17x17 -- one
34x34-patch image after 2x2 merge. So each rank does hold a complementary slice
of the sentinels, and `_select_cp_shard`'s premise holds.

**The call counts do not line up.** `forward` runs several times per step (the
probe shows `pixel_values` of (1,1120,588), (1,1140,588), (1,1156,588),
(1,1092,588) -- different microbatches, different images), and the number of
`_exchange_sentinel_counts` entries differs between ranks within a step. A
collective whose *count* differs across participants hangs exactly like one
whose participants differ.

So the remaining defect is not in which slice a rank takes, but in how many
times the exchange runs per step and in what order relative to its peers. That
points at the interaction between the microbatch loop and a per-forward
collective, which is a different problem from the two already fixed.

Not fixed. The next measurement is a per-rank, per-step count of entries with
the ranks' output serialized (the current probe interleaves, which is why the
counts had to be inferred rather than read).
