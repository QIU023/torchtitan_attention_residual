# The upstream merge breaks EP x TP, and it is a contract change, not a bug to patch

The merge gate did its job: `git merge upstream/main` reported no conflicts, compileall was
clean, `kimi_k3` imported, and the model's own CPU suite passed. Then 2 of 13 matrix cells
failed. Static checks could not have caught this.

## What the gate measured

Merged tree (`merge_upstream_2026_08_12`, upstream at `65fa556be`, 17 commits) against the
pre-merge reference tables, both arms, DEP and dynamic CP on:

| | full-parameter | LoRA |
|---|---|---|
| byte-identical | 7 | 9 |
| differ in the 5th significant digit (1e-5) | 4 | 2 |
| **FAIL** | **2** | **2** |
| maxdeg cells (5) | all pass, byte-identical | all pass, byte-identical |

The same two cells fail in both arms: `ep2_fsdp2_tp2_pp2` and `ep2_fsdp2_tp2_cp2`. Every
cell with EP but no TP passes (`ep2_fsdp2`, `ep2_fsdp2_pp2_cp2`, `ep8_fsdp8`); every cell
with TP but no EP passes (`tp2`, `tp4`, `fsdp2_tp2_cp2`). **The trigger is EP and TP
together.**

    RuntimeError: Can not redistribute from S(1) to P(sum),
                  redistributing to Partial is for internal use only!
      torchtitan/models/common/moe.py:472 forward
      torchtitan/protocols/module.py:705 _redistribute_outputs

## Why it happens

Upstream's new declarative MoE sharding (`#3970` unified the EP token dispatcher API,
`#3996` reworked MoE sharding) routes the MoE forward through
`Module.forward_with_redistribution`, which validates the output against
`out_src_shardings` and redistributes it to `out_dst_shardings`. In
`common/moe_sharding.py::_routed_experts_sharding_configs`:

    experts_output_layout         = seq_parallel if enable_ep else activation(tp=P)  # out_src
    desired_experts_output_layout = seq_parallel if enable_sp else activation(tp=P)  # out_dst

With `enable_ep=True, enable_sp=False` -- exactly K3's configuration -- the source is
sequence-parallel and the destination is Partial, so the code asks DTensor for
`S(1) -> P(sum)`, which DTensor forbids. Upstream's own models never reach this pair
because they pass `enable_sp=parallelism.enable_sequence_parallel` and enable SP whenever
TP is on; K3 hard-codes `enable_sp=False` because its TP plan is imperative.

And the reason SP is not incidental here: `dense_sequence_parallel_placement()` declares
`partition_spec=(DP, (CP, TP), None)` -- the sequence dimension is sharded over the
**flattened (CP, TP) axes**. So when EP is on, the routed-experts output is SP-shaped on the
TP axis BY CONSTRUCTION. The post-merge contract is therefore:

> **With EP and TP both enabled, the MoE boundary is sequence-parallel over (CP, TP).**

K3's `apply_tp_kimi_k3` presents replicated / plain tensors at that boundary instead --
by design, because plain boundaries are what let PP's P2P, AttnRes's `torch.stack` and
fla-core's triton kernels work (see `parallelize.py`'s module docstring). Before the merge
the two coexisted. Now they contradict.

## What was tried and rejected

Keying `desired_experts_output_layout` on `(enable_sp or enable_ep)` instead of
`enable_sp`. It removed the first error and produced the next one, one level up:

    ValueError: MoE: output DTensor has placements (Shard(dim=1),),
                but out_src_shardings expects (Partial...)

i.e. the MoE wrapper's own `out_src_shardings` is keyed the same way. Three sites are keyed
on `enable_sp` where the source layouts are keyed on `enable_ep`, so this is a chain of
assertions, not a one-line fix -- and chasing it assertion-by-assertion is the pattern that
cost five rounds on the vLLM weight-sync path this same day. **The patch was reverted.**

## The two real resolutions

1. **Adopt the declarative path for K3's MoE.** Write `kimi_k3/sharding.py` in the shape of
   `qwen3/sharding.py` and `deepseek_v3/sharding.py`, pass the real
   `parallelism.enable_sequence_parallel`, and make the K3 TP plan SP-consistent at the MoE
   boundary. This is the reuse-over-duplication direction and almost certainly where
   upstream is heading. It is a redesign of the MoE half of `apply_tp_kimi_k3`, and it
   interacts with the plain-boundary requirement that PP and the fla kernels impose.
2. **Stop letting `set_moe_sharding_config` own the TP axis for K3.** Populate only the
   EP-relevant state shardings and keep the imperative plan. Smaller and lower risk, but it
   maintains a divergence from the mechanism upstream now uses, which is a cost every future
   merge pays.

Either way this is a design decision with a matrix run attached, so **the merge branch is
NOT folded into `attention_residual_dev` yet.**

## Consequence for PR-B

`k3_pr_b_ep_grouped`'s scope is EP plus the grouped-GEMM path. Post-merge, the EP x TP
interaction is no longer expressible the way the PR body describes it, so that kit's
extraction list needs revisiting once resolution 1 or 2 is chosen. PR-A (TP) is affected
from the other side.

## The 1e-5 cells, separately

Six cells across the two arms differ only in the last printed digit. Most likely cause is
`#4099 Keep global valid token counts on device`, which moves a reduction from host to
device and therefore changes summation order in the loss denominator. Benign -- but
**"byte-identical to the pre-merge table" is no longer an available claim for the merged
tree**, and any future regression check has to compare against a post-merge baseline.
