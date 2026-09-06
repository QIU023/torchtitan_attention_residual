# Do the MLA CP kernels compose with `overrides/fused_mla.py`? (2026-09-06)

tianyu-l's question on 4313 (`context_parallel.py:70`). The override is specific to DeepSeek-V3's
`Attention` (`@override(Attention.Config)`, needs ComplexRoPE) and does not attach to
`KimiMLAAttention`, so the test runs `deepseek_v3_debugmodel` on the CP tree (`61a73ca6c`, worktree
`/tmp/wt_dsfused` with three probe flavors: the plain model, cp2 on the upstream Ulysses kernel, cp2
on the packed `MLAUlyssesCPFlexAttention` with `rope_head_dim=64`), `ds_fused_mla_cp_probe.sh`.
DeepSeek-V3 is T-major with `k = [k_nope | k_pe.expand(heads)]` like K3, so the packed kernel's
premise holds there. 4096 tokens per step in 256-token micro-batches, seed 42, one seed checkpoint,
3 steps; CUDA graphs off (the token-budget deepseek run refuses capture, as in the pipeline control)
and activation checkpointing off in every cell: with selective AC on, the override fails at its first
forward with "Tensor cached during selective activation checkpoint has been mutated" (the override's
own constraint, independent of CP).

| cell | world | stock (step 1 / 3) | with `--override.imports torchtitan.overrides.fused_mla.fused_mla` |
| --- | --- | --- | --- |
| dp1 | 1 | 8.15954 / 4.87367 | 8.15954 / 4.87367 |
| cp2, generic Ulysses (4450) | 2 | 8.15959 / 4.87362 | 8.15959 / 4.87362 |
| cp2, packed MLA Ulysses (this PR) | 2 | 8.15959 / 4.87365 | 8.15959 / 4.87365 |

Reading: the override changes nothing numerically (each cell bitwise its stock twin through step 3),
and the packed kernel composes with it exactly as the upstream kernel does. Packed against generic
on DeepSeek: step 1 bitwise, step 3 within 3e-5 (the rope-slice addition order, not amplified here:
no random-init MoE router flips on it; K3's cp2 kernels differ by percents at step 3 for that reason).
AC on with AC off, dp1: bitwise.
