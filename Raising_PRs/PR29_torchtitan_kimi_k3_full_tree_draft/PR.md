# PR29 -- Kimi K3, whole tree, DO NOT MERGE

**Target**: `pytorch/torchtitan`, `main`. Draft, and marked DO NOT MERGE in the title.
**Branch**: `QIU023/torchtitan:k3_full_tree_draft` (`70e897f73`)
**Size**: 96 files, +28165 -28 against `upstream/main` `48cb7ae1f`, excluding the vendored
reference tree under `models/kimi_k3_up/` which must be dropped before filing.

**Why this exists alongside the axis PRs.** PR21/22/23 each carry one parallelism axis with
the other two raising, which is what makes them reviewable. That deliberately hides the rest
of the work -- MXFP4 QAT, packed-MXFP4 import, quantile load balancing, MTP, LoRA, DEP, the
MoonEP dispatcher, the HF<->DCP key map. This one is the whole thing, so nobody finds out
about a piece of it late.

The 28 deletions are the point to notice: this is almost entirely additive.

## What is in it, by commit

Sliced by content rather than by history -- the branch's own 381 commits are a working
record, not a review sequence. Fifteen commits:

| # | commit | files |
| --- | --- | --- |
| 01 | the K3 model: KDA, MLA, latent MoE, MTP, and its config tree | `__init__.py` `model.py` `model_configs.py` `config_registry.py` `knobs.py` `README.md` `mtp_loss.py` `models/__init__.py` |
| 02 | Block Attention Residuals: the primitive and the layout tables | `attn_res.py` `attn_res_model.py` `layout.py` |
| 03 | SiTU-GLU routed experts and quantile load balancing | `moe.py` `quantile_balance.py` `common/moe.py` `common/moe_sharding.py` `models/utils.py` |
| 04 | MoonViT and the multimodal wrapper | `moonvit.py` `multimodal_model.py` `vision_preprocess.py` |
| 05 | tensor parallelism, including the KDA DTensor shims | `quant_scope.py` |
| 06 | context parallelism: KCP for KDA, Ulysses for MLA, dynamic CP for the tower | `kcp.py` `vit_cp_plan.py` |
| 07 | pipeline parallelism: the cross-stage AttnRes adapter | `pipeline_adapter.py` |
| 08 | DEP: the vision tower on its own stage, with bubble scheduling | `dep_bubble_{plan,runtime,backward}.py` `vit_prefetch.py` |
| 09 | a MoonEP token dispatcher against torchtitan's EP seam | `moon_ep_dispatcher.py` |
| 10 | MXFP4 QAT and packed-MXFP4 weight import | `mxfp4_qat.py` `packed_mxfp4.py` |
| 11 | LoRA, including the skip-edge gradients PP must route | `lora.py` `muon.py` |
| 12 | HF <-> DCP conversion for the released key set | `hf_key_map.py` `state_dict_adapter.py` |
| 13 | the parallelize entry that applies all of the above | `parallelize.py` |
| 14 | the core changes the above needs | `components/{lr_scheduler,optimizer}.py` `distributed/{fsdp,utils}.py` `tools/grouped_mm_empty_shim.py` |
| 15 | tests for all of it | `tests/` (55 files) |

## Two upstream defaults this tree does not yet satisfy

Both are pinned in the gate rather than worked around in the model, and both are stated
here because a reviewer running the branch with stock defaults will hit them immediately.

**`spmd_backend` defaults to `spmd_types` (#4085), and this tree needs
`partial_dtensor`.** Not a flag we forgot: `fully_shard()` under `spmd_types` requires every
parameter to already be a DTensor on the full SPMD mesh, and TP additionally trips
`assert_type() does not support DTensor. SPMD type checking operates on local tensors
only`. A partially declarative model cannot satisfy either. So supporting `spmd_types` IS
the declarative conversion -- the same work as giving this model a `sharding.py` -- rather
than a patch on top of it. That is the single largest gap between this `parallelize.py` and
an upstream model's, and it is the next thing we do.

**CUDA graph capture is on by default (#3559), and this tree needs
`--training.disable-cuda-graphs`.** The vision path's patch count varies per batch, so
capture validation rejects it: `input 4 changed from (1, 256, 588) to (1, 192, 588)`. This
one is not specific to us -- any VLM with a dynamic patch count reaches it -- so it may be
worth a separate issue rather than a flag.

## Evidence

58 of 58 cells across three model arms (text / multimodal / multimodal+LoRA), on this
branch, with the two pins above. Per-cell loss and grad_norm for all 10 steps are in the
logbook (`gate_logs/gate_58_2026-08-19_merged_percell.txt`) rather than here.

The run's own accounting line says 54, and that number is worth explaining rather than
hiding: the four DEP and pp8xvp4 cells are launched by two scripts that build their own
torchrun line instead of going through the shared one, so the CUDA-graph pin never reached
them and they failed identically on "CUDA graphs do not support pipeline parallelism yet".
Both scripts now take the same pin, and those four pass at 10 steps each -- 9.91233, 9.88840
(DEP pp4/pp8) and 11.86407, 9.54677 (pp8xvp4 lora/mm).

Against the same gate on the pre-merge tree, `fsdp2`, `cp2` and `ep2_fsdp2` are identical at
four steps and `tp2` differs by one in the last digit of step 1 -- so merging 41 upstream
commits moved essentially nothing, which is the claim worth making about a merge.

## Before filing

* drop `models/kimi_k3_up/` (8 files) -- a vendored copy of the other K3 PR's tree, kept
  locally to diff against and not something to send upstream;
* 14 of our own commits carry a bare `#NNNN`, which would fire cross-reference events in
  this repo's timeline on open. The fifteen-commit rebuild is written fresh, so those
  messages do not travel -- but check the rebuilt messages before pushing;
* the title must carry DO NOT MERGE, and the PR must be opened as a draft.
