# Smoke coverage, end of the fix-and-connect phase (2026-07-30)

Everything below was measured on this box (8x RTX 5060 Ti, PCIe, no NVLink), with
every parallelism leg loading a shared seed checkpoint so losses are comparable
rather than merely rank-consistent.

## Connected and verified

| area | evidence |
| --- | --- |
| structure vs official reference | MLA rel 7.07e-06, KDA 2.67e-03, SiTU/LatentMoE/router structurally identical, 2p8t config 29/29 fields |
| weight naming | 497,220 released checkpoint keys mapped, non-expert keys round-trip |
| MXFP4 decode | rel 0.0 against torchao on torchao's own bytes |
| FSDP / PP / CP / EP | 12 of 13 legs pass; EP numerically identical to no-EP |
| PP degree | PP2 and PP4 pass; PP8 blocked (below) |
| VP | VP=1, 2 and 4 all pass, and VP contributes NO numerical difference |
| CP numerics | KCP layer-0 1.8e-05 vs control 7.2e-02 at cp2; conv halo bit-exact at cp2/4/8 |
| multimodal | MoonViT-V2 + backbone forward/backward at dp2, gradients reduce-scattered |
| training recipe | Muon and Quantile Balancing both selectable and measured against AdamW+sign from one seed |
| veRL actor | loads real weights, initializes (80.9M params) |
| QLoRA from official-format weights | loads HF safetensors into a LoRA-wrapped model and trains; step-1 loss 7.71304 EXACTLY matches the full-param dp2 baseline, which is LoRA's step-0 identity and confirms the base weights landed |
| unit tests | 260 passed, 1 skipped, 82 subtests |

## Open, with the reason

| item | status |
| --- | --- |
| **TP** | Unattributed one-directional gradient gap (3.10x end to end, ~4.7%/layer). Three hypotheses tested and retracted; see TP_GRAD_FINDING. Set aside by decision, not resolved. |
| **PP8** | Blocked by a framework/hardware interaction, not a PP defect: dp_shard=1 + PP means mixed precision applies via neither fully_shard nor autocast, so the model runs fp32, and fp32 KDA needs 108,160 B against this GPU's 101,376 B. Needs 16 GPUs or a 227 KB datacenter part. |
| **SFT** | No recipe exists and no instruction dataset is available here. Never verified -- writing one would be a new feature, not a verification. |
| QLoRA from a DCP seed | Not supported; the DCP path validates keys before module hooks run. Deliberately not built: the real workflow is official HF weights plus adapters, which works, and the DCP variant was only ever a harness convenience. |
| MoonEP | Needs one 8-GPU NVLink node (cuMulticast unsupported here). |
| official 2.8T weights | 1.561 TB; needs 16xH200-class memory. Load logic verified at k3mini scale. |
| multimodal training | needs an image dataset (obelics streaming). |

## Method notes this phase paid for

Four wrong conclusions were published and retracted in this phase, all from the
same root cause: **measuring in a regime where the effect under test cannot appear,
then claiming generality.**

* "TP grad_norm is a metric artifact" -- disproven by measuring the trainer's own
  clip_grad_norm_ against the materialized norm (they agree exactly).
* "TP gradients are systematically attenuated" -- downgraded once the whole-model
  ratio was shown to saturate, because this model amplifies perturbations ~1.6x per
  layer.
* "the MLA output gate carries it" -- the ablation compared two different models,
  and the proposed mechanism was wrong (x arrives at the gate as a DTensor, so
  DTensor already redistributes that gradient).
* "PP is numerically transparent" -- true only at one microbatch, which is where it
  is trivially true; at 8 microbatches PP differs by a bf16 accumulation margin.

The working rules that came out of it: a control that scores close to the arm under
test means the metric has saturated and neither number means anything; a leg
reported as failing must be reproduced in isolation before it is believed; and an
ablation that deletes a module does not isolate that module's parallelism handling.
