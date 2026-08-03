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
    fsdp2_pp2_cp2              see below

    ep2_fsdp2                  7.67856 7.26963 6.29357
    ep2_fsdp2_tp2_pp2          7.73785 7.21456 6.31275
    ep2_fsdp2_tp2_cp2          7.71987 7.22920 6.35607
    ep2_fsdp2_pp2_cp2          see below

## The two PP+CP combinations produce no loss at all

Both configurations containing pp2 together with cp2 and fsdp2 run to completion
and print nothing. Not a harness artifact and not a crash:

- exit code 0, "Training completed" on every rank
- the trainer initializes correctly (local batch 2, global 8, gradient
  accumulation 2, 3 steps)
- grepping every rank's output for `loss:` -- including the negative
  placeholders that PP stages not holding the loss emit -- returns nothing

So the run trains and reports no loss from any rank. Recorded as a PRE-EXISTING
defect: it is there before the refactor, so the refactor cannot be blamed for it,
and it must not be counted as a passing leg.

Noted from the same log and independent of this: FSDP2 warns that
`FSDPKimiAttnResDecoderLayer` returns a view tensor, and that an in-place op on a
view silently drops the pre-backward hook and skips the all-gather, which "can
cause backward to fail or produce wrong gradients". AttnRes returns aggregated
views by construction, so that deserves its own look.

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
