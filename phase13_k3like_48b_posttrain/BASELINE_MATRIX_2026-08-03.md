# Full D matrix baseline, before the config-tree refactor

`kimi_linear_k3mini_diag_4l_moe_depth`, 3 steps, seed 42, deterministic, global
batch 8. Every refactor step has to reproduce these before moving on.

    dp1                        7.72376 7.14969 6.22981
    fsdp2                      7.67856 7.26904 6.29599
    pp2                        7.66117 7.04581 6.12275
    cp2                        7.69600 7.21146 6.26816
    tp2                        7.70027 7.04398 6.15859

    fsdp2_tp2_pp2              7.71036 7.23313 6.28071
    fsdp2_tp2_cp2              7.76396 7.31618 6.39315
    tp2_pp2_cp2                7.71961 7.15782 6.29697
    fsdp2_pp2_cp2              7.70254 7.23634 6.33279

    ep2_fsdp2                  7.67856 7.26963 6.29357
    ep2_fsdp2_tp2_pp2          7.73785 7.21456 6.31275
    ep2_fsdp2_tp2_cp2          7.71987 7.22920 6.35607
    ep2_fsdp2_pp2_cp2          7.75924 7.24905 6.52276

## Correction: those two legs were never broken -- the collector was

The first run of this matrix reported the two PP+CP legs as producing no loss,
and called it a pre-existing defect. That was wrong, and the way it was wrong is
worth keeping.

The runs did print their loss. Under PP the stages that do not hold the loss emit
a negative placeholder (-8.00000 here, -2.00000 under pp2 alone, i.e.
-global_batch_size), so the collector filtered negatives to drop them. But it
filtered AFTER `sed` had rewritten each line to "<step> <loss>" and was
deduplicating with `sort -k1,1n -u`, which keeps ONE row per step -- and the row
it kept was the placeholder. Dropping negatives then removed the only surviving
row for that step, leaving nothing.

Deduplicating on the pair instead of the step, after dropping placeholders,
recovers them:

    fsdp2_pp2_cp2       7.70254 7.23634 6.33279
    ep2_fsdp2_pp2_cp2   7.75924 7.24905 6.52276

So the matrix is 14/14, and there is no pre-existing PP+CP defect. The lesson is
the same one this project keeps re-learning: a filter that silently removes
everything looks identical to a failure, and "no output" should be diagnosed
before it is characterized.

The FSDP2 warning noted from the same log stands on its own and is unaffected:
`FSDPKimiAttnResDecoderLayer` returns a view tensor, and an in-place op on a view
silently drops the pre-backward hook and skips the all-gather. AttnRes returns
aggregated views by construction, so that still deserves its own look.

## Why this refactor, and what it is NOT for

Upstream models are config-tree constructed -- deepseek_v3 declares wq_a / wkv_a
/ wo as `Linear.Config`, llama3 builds from `config.attention` -- and both
inherit `Decoder`. `LoRAConverter.convert()` starts from
`model_config.traverse(Module.Config, recurse=True)`, so it needs that tree. K3
builds its linears positionally on a plain `nn.Module`, with no tree to walk,
which is why the converter cannot be adopted.

NOT for 2.8T scale-up. `kimi_linear_k3_2p8t_block_attn_res` already builds
config-only at 93 layers / hidden 7168 / 96 heads / 896 experts top-16, through
the same `_flavor_trainer_config` parameterization as every other size. That goal
is already met and should not be used to justify this work.

NOT for migrating TP to the declarative path either -- k3_refactor established
that the declarative vocabulary has no `use_local_output` and so cannot express
K3's plain-tensor module boundaries.

## Refactor stopped: upstream PR #4025 removes its last justification

pytorch/torchtitan#4025 adds Kimi K3 upstream. Checked against the three reasons
this branch existed:

- It constructs `nn.Linear` POSITIONALLY, not through a config tree. It has
  config dataclasses but does not declare child `Linear.Config` fields or
  `.build()` them. So "return to the titan standard" was never true -- upstream's
  own K3 does what ours does.
- It supports **FSDP2 only**, and explicitly rejects HSDP, TP, PP, CP, EP,
  activation checkpointing, torch.compile and CPU offload. The author notes
  TP/PP/CP would need significant adaptation because of data-dependent Python
  loops and incompatible forward signatures.

That inverts the argument. The parallelism work upstream declines to do is
exactly what this fork has: 14/14 matrix legs producing loss, PP verified
per-parameter at 0.00000 over 548 parameters, two TP defects found and fixed
(one of which also fixes upstream deepseek_v3). Refactoring toward a style
upstream does not use, at the cost of breaking that, is the wrong trade.

Measured cost, not estimated: converting three MLA linears to `Linear.Config(
...).build()` failed 12 of 14 legs with silent `exit=0` hangs. Reverted.

The branch keeps its two gated commits (the MLA Config declaration and the
sharding declarations, both bit-identical on the matrix) in case the
LoRAConverter question returns. It is not the current line of work.
