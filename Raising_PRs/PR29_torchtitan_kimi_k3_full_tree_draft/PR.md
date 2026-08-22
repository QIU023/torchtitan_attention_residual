# PR29 -- Kimi K3, whole tree, DO NOT MERGE

**Target**: `pytorch/torchtitan`, `main`. Draft, and marked DO NOT MERGE in the title.
**Branch**: `QIU023/torchtitan:k3_pr_classified_v2` at `f0f2e6986` -- 15 commits on `upstream/main`,
sliced by content. `k3_full_tree_draft` is where the merge and the adaptations were done and
is not what gets filed.
**Size**: 98 files, +28307 -28 against `upstream/main` `00cffaeb3`. The vendored reference
tree under `models/kimi_k3_up/` is gone, and so is its entry in `models/__init__.py`.

**Why this exists alongside the axis PRs.** PR21/22/23 each carry one parallelism axis with
the other two raising, which is what makes them reviewable. That deliberately hides the rest
of the work -- MXFP4 QAT, packed-MXFP4 import, quantile load balancing, MTP, LoRA, DEP, the
MoonEP dispatcher, the HF<->DCP key map. This one is the whole thing, so nobody finds out
about a piece of it late.

The 28 deletions are the point to notice: this is almost entirely additive.

## What is in it, by commit

Sliced by content rather than by history -- the branch's own 409 commits are a working
record, not a review sequence. Fifteen commits:

| # | commit | files |
| --- | --- | --- |
| 01 | the K3 model: KDA, MLA, latent MoE, MTP, and its config tree | `__init__.py` `model.py` `model_configs.py` `config_registry.py` `knobs.py` `README.md` `mtp_loss.py` `models/__init__.py` |
| 02 | Block Attention Residuals: the primitive and the layout tables | `attn_res.py` `attn_res_model.py` `layout.py` |
| 03 | SiTU-GLU routed experts and quantile load balancing | `moe.py` `quantile_balance.py` `common/moe.py` `common/moe_sharding.py` `models/utils.py` |
| 04 | MoonViT and the multimodal wrapper | `moonvit.py` `multimodal_model.py` `vision_preprocess.py` |
| 05 | the quantization scope -- what K3 quantizes, shared by QAT and QLoRA | `quant_scope.py` |
| 06 | context parallelism: KCP for KDA, Ulysses for MLA, dynamic CP for the tower | `kcp.py` `vit_cp_plan.py` |
| 07 | pipeline parallelism: the cross-stage AttnRes adapter | `pipeline_adapter.py` |
| 08 | DEP: the vision tower on its own stage, with bubble scheduling | `dep_bubble_{plan,runtime,backward}.py` `vit_prefetch.py` |
| 09 | a MoonEP token dispatcher against torchtitan's EP seam | `moon_ep_dispatcher.py` |
| 10 | MXFP4 QAT and packed-MXFP4 weight import | `mxfp4_qat.py` `packed_mxfp4.py` |
| 11 | LoRA, including the skip-edge gradients PP must route | `lora.py` `muon.py` |
| 12 | HF <-> DCP conversion for the released key set | `hf_key_map.py` `state_dict_adapter.py` |
| 13 | the parallelize entry that applies all of the above, incl. the TP plan and the KDA DTensor shims | `parallelize.py` |
| 14 | the core changes the above needs | `components/{lr_scheduler,optimizer}.py` `distributed/{fsdp,utils}.py` `tools/grouped_mm_empty_shim.py` |
| 15 | tests for all of it | `tests/` (56 files) |

## Two upstream defaults this tree does not yet satisfy

Both are pinned in the gate rather than worked around in the model, and both are stated
here because a reviewer running the branch with stock defaults will hit them immediately.

**`spmd_backend` defaults to `spmd_types` PR-4085, and this tree needs
`partial_dtensor`.** Not a flag we forgot: `fully_shard()` under `spmd_types` requires every
parameter to already be a DTensor on the full SPMD mesh, and TP additionally trips
`assert_type() does not support DTensor. SPMD type checking operates on local tensors
only`. A partially declarative model cannot satisfy either. So supporting `spmd_types` IS
the declarative conversion -- the same work as giving this model a `sharding.py` -- rather
than a patch on top of it. That is the single largest gap between this `parallelize.py` and
an upstream model's, and it is the next thing we do.

**CUDA graph capture is on by default PR-3559, and this tree needs
`--training.disable-cuda-graphs`.** The vision path's patch count varies per batch, so
capture validation rejects it: `input 4 changed from (1, 256, 588) to (1, 192, 588)`. This
one is not specific to us -- any VLM with a dynamic patch count reaches it -- so it may be
worth a separate issue rather than a flag.

## Evidence

58 of 58 cells across three model arms (text / multimodal / multimodal+LoRA), with the two
pins above, on `afc3e4287` -- the merged tree carrying the 2026-08-21 embedding and
grad-norm fixes, which is zero files different from the re-cut `k3_pr_classified_v2` once
the vendored tree is excluded. Per-cell loss and grad_norm for all 10 steps:
`gate_logs/gate_58_2026-08-21_final_percell.txt`. TP cells sit at grad_norm 3.2-3.5 there;
the 08-19 log's 12-18 was the inflation, fixed the same day
(`TP_GRADNORM_INFLATION_2026-08-21.md`).

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

* ~~drop `models/kimi_k3_up/`~~ -- done, twice over. v1 removed it by amending commit 01;
  `k3_pr_classified_v2` is a re-cut on `00cffaeb3` with the registration removed inside the
  model commit where that file belongs. Verified on v2: zero `kimi_k3_up` files, zero
  registry references. The three axis branches still carry the pre-fix numerics and get
  re-cut after PR-4025, not now.
* ~~bare `#NNNN` in commit messages~~ -- v2 scans clean at 0 of 15, re-verified after the
  re-cut. Nothing rewrites anyone else's commits.
* ~~commit 05's title does not match its content~~ -- fixed by a message-only rewrite,
  trees byte-identical (verified: `git diff` against the pre-rewrite head is empty). 05 is
  now "the quantization scope -- what K3 quantizes, shared by QAT and QLoRA" and 13 picks
  up the TP plan and the KDA DTensor shims it actually contains. Force-pushed; the branch
  head is `f0f2e6986`. All fifteen titles re-scanned against their files: no other
  mismatch.
* the title must carry DO NOT MERGE, and the PR must be opened as a draft.

## PASTE

Everything above is ours. What follows is the PR description, verbatim. Paragraphs are single
lines on purpose -- GitHub reflows them, and hard-wrapped source is what the PR-text rule calls out.

--- PASTE BEGIN ---

Draft, and not for merge. The three axis PRs each carry one parallelism with the others raising, which is what makes them reviewable, and that deliberately leaves out most of the work. This is the whole Kimi K3 tree so none of it is a surprise later: MXFP4 QAT and packed-MXFP4 import, quantile load balancing, MTP, LoRA, the decoupled vision encoder, a MoonEP token dispatcher, and the HF <-> DCP key map for the released checkpoint.

PR-4025 is a separate implementation of the same model and is further along on the model itself. When it lands, this rebases onto it and what remains is the parallelism and the post-training pieces. Filing now is disclosure, not a request to review 28k lines.

98 files, +28307 -28. The 28 deletions are the thing to notice -- this is almost entirely additive, and the five core files it does touch are listed in commit 14.

The fifteen commits are sliced by content rather than by the 409 commits of history that produced them:

| # | commit | main files |
| --- | --- | --- |
| 01 | the K3 model: KDA, MLA, latent MoE, MTP, and its config tree | `model.py` `model_configs.py` `config_registry.py` |
| 02 | Block Attention Residuals: the primitive and the layout tables | `attn_res.py` `attn_res_model.py` `layout.py` |
| 03 | SiTU-GLU routed experts and quantile load balancing | `moe.py` `quantile_balance.py` `common/moe.py` |
| 04 | MoonViT and the multimodal wrapper | `moonvit.py` `multimodal_model.py` `vision_preprocess.py` |
| 05 | the quantization scope -- what K3 quantizes, shared by QAT and QLoRA | `quant_scope.py` |
| 06 | context parallelism: KCP for KDA, Ulysses for MLA, dynamic CP for the tower | `kcp.py` `vit_cp_plan.py` |
| 07 | pipeline parallelism: the cross-stage AttnRes adapter | `pipeline_adapter.py` |
| 08 | the decoupled vision encoder, with bubble scheduling | `dep_bubble_*.py` `vit_prefetch.py` |
| 09 | a MoonEP token dispatcher against torchtitan's EP seam | `moon_ep_dispatcher.py` |
| 10 | MXFP4 QAT and packed-MXFP4 weight import | `mxfp4_qat.py` `packed_mxfp4.py` |
| 11 | LoRA, including the skip-edge gradients PP has to route | `lora.py` `muon.py` |
| 12 | HF <-> DCP conversion for the released key set | `hf_key_map.py` `state_dict_adapter.py` |
| 13 | the parallelize entry that applies all of the above, incl. the TP plan and the KDA DTensor shims | `parallelize.py` |
| 14 | the core changes the above needs | `distributed/{fsdp,utils}.py` `components/{lr_scheduler,optimizer}.py` |
| 15 | tests | `tests/` (56 files) |

Two upstream defaults have to be overridden to run it, both pinned in our gate rather than worked around in the model. `spmd_backend` needs `partial_dtensor`: under `spmd_types`, `fully_shard()` wants every parameter to already be a DTensor on the full SPMD mesh, and TP additionally trips `assert_type() does not support DTensor`, so supporting it is the declarative conversion itself rather than a patch on top -- that is the largest gap between this `parallelize.py` and an upstream model's, and it is what we are doing next. And CUDA graph capture needs `--training.disable-cuda-graphs`, because the vision path's patch count varies per batch and capture validation rejects it; that one is not specific to this model, so it may deserve its own issue.

Evidence: 58 of 58 gate cells pass on this tree, ten steps each, with those two pins. Per-cell loss and grad_norm for every step of all 58 are here, one line per cell:

https://github.com/QIU023/torchtitan_attention_residual/blob/f81a19319506ef1c3d3e2b8fc6eadbf7e6d99feb/phase13_k3like_48b_posttrain/gate_logs/gate_58_2026-08-21_final_percell.txt

The three model arms are text, multimodal, and multimodal+LoRA, 18 cells each, over dp / fsdp / tp / cp / pp / ep and their combinations up to `ep2_fsdp2_tp2_cp2`; the remaining four are the two decoupled-encoder cells at pp4 and pp8 and the two pp8xvp4 cells.

--- PASTE END ---

## Update comment (rolling; post after the TP-migration push lands on the branch)

--- PASTE BEGIN ---

Rebased onto current upstream main and re-validated: 58 parallelism configurations -- FSDP2/HSDP, TP, PP (incl. interleaved 8x4), CP, EP and their combinations, each across a text, a multimodal, and a multimodal+LoRA debug model -- train 10 steps under `--debug.seed 42 --debug.deterministic`. The TP plan moved from an imperative plan to sharding configs declared on the modules, following deepseek_v3 (apply_tp 428 -> 374 lines); loss and grad_norm are bit-identical before vs after in every configuration without LoRA. LoRA configurations move because adapter sharding now derives from the base layer's declaration.

Pending final evaluation:
- MXFP4 QAT, packed-MXFP4 weight import, quantile expert balancing
- MoonEP token dispatcher -- interface draft; needs NVLink hardware (a 2-GPU NVLink pair is enough)

Still to do:
- finish the TP declarative migration (per-layer AttnRes norms, MoonViT, packed-MXFP4 remain imperative)
- remove the module-boundary unwrapping (use_local_output / to_local)
- rebase onto the reference-model PR when it lands

--- PASTE END ---
