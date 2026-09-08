# verl: what the fork carries and what can go upstream (2026-09-08)

Fork `QIU023/verl`, branch `kimi_k3_integration_rebased` (`3dfd0430`), 41 commits over verl-project/verl main `7cb65014`. Upstream main pins torchtitan nightly `0.1.0.dev20260701` (its engine imports `torchtitan.components.checkpoint` and `lr_scheduler`, both gone from torchtitan main), and its torchtitan engine raises `NotImplementedError` under pipeline parallelism, has no CP padding and no TP path. Upstream already gathers sharded expert stacks whole before the HF split (#7324), so that fix from the fork is not needed there.

## Ready now

| branch on the fork | content | state |
| --- | --- | --- |
| `sync_fixes_upstream` (`e57a94e0`) | `training/rollout_logprobs_diff_max/mean`, the log-prob-scale rollout-vs-actor metric | one file, +6; body in `Raising_PRs/PR_verl_engine_sync/PR_BODIES.md`; raise any time |

## Parked, with the reason

| branch | content | waits for |
| --- | --- | --- |
| `qat_sync_upstream_pending` (`19d13c57`) | fake-quantized expert stacks in the sync for QAT models, unit test | verl bumping its torchtitan pin past July, and the QAT converter (`mx_qat`) landing in torchtitan (the K3 QAT PR carries it); inert until then |

## The engine's parallelism support (the fork's real value; a project, not a PR)

The fork's torchtitan engine runs Kimi K3 GRPO on every parallel axis with the rollout-actor log-prob gap at the cross-engine floor (fsdp2, ep2, tp2, cp2, pp2, the 8-GPU fsdp2 x ep2 x cp2 x pp2, tp2 x cp2 x pp2; `VERL_PARITY_2026-09-07.md`). Upstream has none of the PP / CP / TP paths. What that would take:

1. verl bumps its torchtitan pin to a current nightly (the import paths and `ParallelDims` API moved); the fork's engine already targets current torchtitan.
2. Generic engine commits, in dependency order (each a PR of its own once 1 is done): CP over a packed no-padding stream (`b2d0cd02`, `60d21238`), the pipeline schedule bridge (`3963341f`, `aa91b776`, `0d8a5ae5`, `3becd138`, `92dd4d21`), the mesh lookup without an fsdp mesh (`4a93e121`), TP padding and vocab gather (`14e5e0d4`), every pipeline rank ships every stage in the sync (`46e40b11`), the FSDP2 grad-upcast guard under PP (`994a50f8`), config-registry flavors (`ec0b4354`, `709d37c4`).
3. Model-specific commits stay in the fork until Kimi K3 is in verl's model table: the folded-stream input contract, `cu_seqlens` for KDA (`4b08b917`), the processor/tokenizer acceptance, the LoRA naming for K3's LoRA wrappers.
4. Diagnostics (`b1523b2f`, `e8af3dd2`, the `KIMI_GRPO_*` switches) do not go upstream.

Order of value for a verl contribution: the pin bump plus CP (small, generic, exercised by our cells), then PP (the largest piece and the one upstream explicitly defers), then TP.

## Numbers upstream reviewers will ask for

Step-1 rollout-vs-actor log-prob gap on the Kimi K3 debug export, every cell at the floor 0.10-0.12 after the sync fixes; the QAT cell at 0.26 with the residual attributed (weights 0.47, activations 0.24 offline). Tables and probes: `VERL_PARITY_2026-09-07.md`, `matrix_scripts/verl_parity/`.
