# Full-folder code review: torchtitan/models/kimi_k3 (2026-08-09)

Twelve-angle deep review (5 line-by-line correctness scans by subsystem,
removed-behavior audit, cross-file tracer, reuse, simplification, efficiency,
altitude, conventions) calibrated against the maintainer rubric observed on the
eager PR's review (reuse-over-reimplementation, slop, test hygiene, core
footprint). Reviewed at fork `eb30ceb22` (post upstream merge, models/kimi_k3
layout); the box's parallel commits (`6664bef71`, `3d67e3bb5`) may have moved
some line numbers and possibly fixed overlapping items -- check each against
HEAD before fixing. Numbering matches the session relay. Items confirmed by
two or more independent angles are marked [xN].

Verification status: finder output consolidated and hand-deduped; the
automated verification pass was still running when this file was cut and its
deltas will be folded in when it lands.

---

## TIER 1 -- Official-weights load path (blocks the "real 2.8T/48B weights" claim)

The from_hf direction is roughly half an inverse of to_hf. Fix as one cluster;
test by round-tripping to_hf -> from_hf on a K3-shaped multimodal flavor AND by
loading a real official shard.

- **#37** `state_dict_adapter.py:403` -- from_hf never strips the
  `language_model.` wrapper prefix; every text tensor of a multimodal
  checkpoint (incl. our own to_hf export) silently dropped -> near-empty load.
- **#72** `state_dict_adapter.py:341` -- expert regex anchored at
  `model.layers.` likewise; stacked `w1_EFD/w2_EDF/w3_EFD` never reassembled
  for multimodal keys.
- **#39** `state_dict_adapter.py:437` -- from_hf lacks to_hf's K3-layout
  delegation: shared experts mapped onto a module path that does not exist on
  the latent path; latent/AttnRes keys silently dropped.
- **#71** `state_dict_adapter.py:411` -- to_hf renames `attn_gate_proj ->
  g_proj`; from_hf has no inverse; official gated-MLA load errors or leaves
  gates at random init.
- **#42** `hf_key_map.py:166` -- g_proj disambiguation compares 0-based
  checkpoint layer index against the folder's 1-based `kda_layers` with no
  normalization helper; every gate misclassified on the natural call.
- **#24** `packed_mxfp4.py:98` -- E8M0 decode deviates from OCP (0x00 -> 0
  instead of 2^-127; 0xFF -> inf instead of NaN); own round-trip masks it.
  Likely fixed for free by #56 (reuse the DCP quantized reader).
- **#41** `model_configs.py:464` -- block-size -> num_blocks -> layers_per_block
  is not idempotent: k3mini's block size 12 over 21 layers trains an 11+10
  partition, not the documented 12+9, and exports `attn_res_block_size=11`.
  Also invalidates every "K3 block size 12 exercised" doc sentence.
- **#62** `vision_preprocess.py:143` -- bicubic downscale without
  `antialias=True` vs the reference antialiased resampler: systematic
  preprocessing divergence, parity/finetune risk, no error.

## TIER 2 -- PP adapter / parallelism interaction (the crown-jewel files)

- **#65** `pipeline_adapter.py:805` -- eval microbatches populate the
  RankLocalCache but eviction is wired only to backward; `pp_schedule.eval()`
  never wrapped -> the next train step reads stale eval blocks under reused
  chunk ids: silently wrong attention/loss after any validation pass.
- **#66** `pipeline_adapter.py:1676` -- adapter install/fallback decided from
  rank-LOCAL state; under DEP the ViT rank and text ranks can disagree on
  whether the delta protocol is active -> first-hop P2P shape mismatch/hang.
  Needs a globally consistent decision.
- **#67** `layout.py:273` -- layer->stage discovery structurally dead under
  multi-rank PP (each rank sees only its own V stages, the != n_layers check
  always fails, silent fallback to contiguous default). Uneven splits then
  mis-assign producers: mid-step assert, or silently dropped skip-edge grads.
- **#31/#68** `pipeline_adapter.py:1155` -- dead twin path derives
  layers_per_block with floor vs the model's ceil; K3-faithful 93/8 raises in
  BlockLayoutTables, blanket `except Exception` converts it into a silent
  fallback to naive PP -- cache off on exactly the production topology, and
  any measurement from such a run is invalid. (Resolves with #25 deletion;
  audit the live path's blanket except too.)
- **#34** `parallelize.py:1484` -- compile carve-out ineffective under TP:
  instance-bound shim forwards (bound before compile) shadow the class-level
  `compiler.disable`; dynamo traces the un-disabled forward.
- **#69** `parallelize.py:1481` -- `torch.compiler.disable(op)` return value
  discarded for chunk_kda/fused_recurrent_kda/fused_kda_gate: disable does not
  mutate in place, the carve-out for the fla entry points is a no-op.
- **#70** `pipeline_adapter.py:1026` -- step-end cache sweep wrapped in bare
  `except Exception: pass`; eviction failures are silent, memory grows until a
  far-away OOM.
- **#32** `pipeline_adapter.py:1446` -- pipeline topology controlled by env
  vars (KIMI_VIT_DEP/_STAGES, TORCHTITAN_ATTNRES_CACHE, KIMI_VIT_TP_HEADS,
  KIMI_VIT_DYNAMIC_CP, KIMI_VIT_SIDE_STREAM): non-uniform launcher export =
  different topologies per rank, hang with no pointer to the cause; runs not
  reproducible from config/checkpoint. Upstream will not accept env-var
  topology -- migrate to config fields before PR-C.
- **#50** `pipeline_adapter.py:1296` -- DEP vision-chunk detection: zero
  markers wired only logs; missing `if wired == 0 and dep_vision_stages() > 1:
  raise` -> silent no-vision run.
- **#51** `vit_prefetch.py:146` -- the unconditional issue-time stream join
  was removed; an aborted step leaves un-joined side-stream collectives ->
  allocator reuse of still-written buffers + the cross-communicator cyclic
  wait the module docstring calls non-negotiable.
- **#61** `vit_prefetch.py:185` -- the concrete deadlock mechanism: side-stream
  encode contains NCCL collectives; host issue order does not order device
  execution across streams -> classic two-communicator cyclic wait.
- **#49** `multimodal_model.py:1253` -- deleted construction-time guard for
  body/tail `step_inputs`; missing grid silently substitutes a placeholder and
  slices real patch payloads; stale docstring contradicts the new assumption.

## TIER 3 -- Published-numbers contamination

- **#45/#53/#73 [x3]** `model.py:2054` -- get_nparams_and_flops MoE patterns
  (`.moe.router` etc.) never match actual FQNs (`ffn._moe....`): every MoE
  param bucketed dense, ALL experts counted activated -> logged MFU/TFLOPs
  inflated ~num_experts/top_k (10-30x) for every MoE flavor, historically and
  today. Fix + decide whether any public MFU number needs retraction. Deeper
  fix per #53: adapt `models/utils.get_moe_model_nparams_and_flops`.

## TIER 4 -- QAT / LoRA / optimizer correctness

- **#19** `lora.py:427` -- NF4 forward drops base bias (mxfp4 and unquantized
  branches add it); biased projections (attn_gate_proj) silently shifted.
- **#20** `lora.py:407` -- `_forward_packed_tp` drops base bias; tp>1 vs tp=1
  numerics diverge for biased projections.
- **#23** `lora.py:880` -- merge on packed base mixes plain dequant tensor
  with DTensor adapters: error, or rank-local shard exported under full-tensor
  key -> corrupted HF export.
- **#22** `mxfp4_qat.py:94` -- no `.weight`/`.bias` passthrough on
  MXFP4QATLinear: `tag_per_head_muon` silently skips all wrapped MLA
  projections (Per-Head Muon degenerates, warning suppressed); checkpoint keys
  shift to `.base.weight`.
- **#46** `moe.py:78` -- SiTU experts call `torch._grouped_mm` directly,
  bypassing the overridable `self._grouped_mm` seam; torchao MX/Float8 expert
  converters silently never invoked, contradicting the file's own docstring.
- **#21** `muon.py:296` -- AdamW fallback group applies weight_decay=0.1 to
  norms/biases/dt_bias/A_log; norm scales decay ~45% over 20k steps. Exempt
  1-D params.

## TIER 5 -- MTP (new port, not yet matrix-covered)

- **#43** `attn_res_model.py:687` -- init_weights skips MTP-nested AttnRes
  layers: gated alphas stay torch.empty garbage after meta->to_empty->init ->
  step-0 NaN; ungated violates the paper's zero-init.
- **#44** `attn_res_model.py:599` -- MTP branch runs before the _skip_lm_head
  early-return: full-vocab logits materialized despite chunked loss (OOM at
  8192), chunk/label misalignment, ~1.3 GiB/depth retained across steps.
- **#48** `attn_res_model.py:602` -- MTP input double-normed
  (hnorm(norm(h)) vs reference hnorm(h_pre_norm)): official-MTP-weight parity
  silently broken.
- **#40** `config_registry.py:1198` -- MTP flavor's loss configured with
  global_vocab_size=163840 against a 2016-wide model.

## TIER 6 -- Vision / dynamic CP correctness

- **#59** `moonvit.py:448` -- gather-KV prefix mask assumes trailing padding;
  video (t>1) padding is per-frame interleaved on deficit ranks -> pad keys
  admitted into softmax, real later-frame keys masked; silently wrong encoder
  output.
- **#60** `multimodal_model.py:1507` -- the DEP single-vision-stage path (the
  one `_dep_reject_cp` directs CP users to) calls the sentinel exchange but
  discards the result and never selects the CP shard -> step-1 ValueError or
  misdispatch.
- **#63** `multimodal_model.py:1161` -- num_images captured before CP shard
  flattening; coincidental sentinel counts feed `_splice` 1-D rows.
- **#64** `vision_preprocess.py:203` -- pack_video records only frame 0's
  (h, w), no same-resolution enforcement; odd-sized frames silently corrupt
  cu_seqlens for the whole batch tail.
- **#47** `model.py:1721` -- base-model multimodal path still uses
  advanced-index assignment (the exact bug fixed in the AttnRes subclass):
  mixed batches / PP shape inference crash.
- **#35** `multimodal_model.py:1585` -- import-time setattr grafting of
  KimiK3Model methods onto the wrapper; grafted init_weights does not init the
  MoonViT tower/projector (silent stale init for train-from-scratch VL).

## TIER 7 -- Reuse (the dominant maintainer rubric)

- **#54** `parallelize.py:1265` -- apply_fsdp is a ~180-line fork of
  `distributed.fsdp.apply_fsdp_to_decoder` and already lags it (no
  enable_symm_mem, no full_dtensor dp_mesh_dims, no Shard(1) refinement, no EP
  prefetch wiring). Converge: shared helper + thin attn_res_tail delta.
- **#56** `packed_mxfp4.py:54` -- MXFP4 decode duplicates torch DCP's
  `QuantizedHuggingFaceStorageReader._dequantize_tensor_mxfp4` (gpt_oss path);
  our `get_hf_storage_reader` still raises for quantized. Reuse likely fixes
  #24 for free.
- **#55** `moonvit.py:499` -- per-image Python SDPA loop vs common
  VarlenAttention / FlexAttention + vision block mask (qwen3_5 vision path).
- **#58** `model.py:385` -- KimiMLAInnerAttention duplicates common
  `ScaledDotProductAttention` (loses backend selection + CP-compatible
  signature).
- **#57** `muon.py:152` -- hand-rolled AdamW clone vs torch.optim functional
  AdamW for fallback params.
- **#6** (manual pass) KimiMLAAttention vs upstream DSv3 MLA: prepare the
  difference table (NoPE / no q compression / broadcast k_rot / gate) with a
  hard "why not parameterize theirs" answer, or converge -- the maintainer
  asked the eager PR exactly this.
- **#7** (manual pass) KimiMLP vs common FeedForward for dense layer + shared
  experts ("should use our fused feed forward" was said verbatim on the eager
  PR).

## TIER 8 -- Flavor / registry / dead code

- **#38** `model_configs.py:136` -- the "mini" alias only aliases the sweep
  row: model_registry-built `kimi_k3_mini_*` is a different architecture
  (163840 vocab, silu, 32 experts...) from the same-named trainer flavor ->
  the veRL seed-checkpoint mismatch class.
- **#74** `__init__.py:205` -- module-scope flavor discovery imports
  model.py, defeating the fla-core guard: CPU-only import crashes; the
  deferred-raise is dead code.
- **#25/#33** `pipeline_adapter.py:1095` -- dead twin pipelining fn +
  helpers (~110 duplicated lines, zero callers, already diverged once): delete.
- **#26/#75 [x2]** `model.py:1931/2153` -- `dim` and `vocab_size` defined
  twice in the same class body; later silently shadows earlier.
- **#29** `multimodal_model.py:105` -- legacy LLaVA scaffold retired (only its
  own test reaches it); deleting it also removes #17's .item() double loop.
- **#28** `model.py:257` -- `is_mla` constant-True by construction; flatten
  dispatch, delete unreachable NotImplementedError.
- **#27** `config_registry.py:149` -- 12 copy-paste flavor stubs -> loop
  generation; nothing checks stubs agree with SCALING_LAW_TABLE.
- **#36** `__init__.py:130` -- magic suffix parsing (already hid 37 flavors
  once); move to structured fields on registry entries.
- **#30** `config_registry.py:489` -- zero-field dataclasses.replace wrapper +
  duplicate import aliases.
- **#76** stale `torchtitan/experiments/kimi_k3/__pycache__` orphan: remove.

## TIER 9 -- Efficiency

- **#13** `attn_res.py:69` -- stack rebuild (+ fp32 copy, verify) every call
  on the hottest path; persistent bank buffer or per-block accumulation.
- **#14** `muon.py:143` -- per-head sequential Newton-Schulz -> batch to 3-D.
- **#15** `moonvit.py:500` -- `cu_seqlens.tolist()` host sync per block x27,
  serializes the DEP side stream; hoist once.
- **#16** `quantile_balance.py:366` -- one all_reduce per MoE layer (~92
  sequential collectives/step) -> stack + single all-reduce.
- **#18** `attn_res_model.py:512` -- per-microbatch `.any()` host sync on the
  stage-0 embed path; vectorize unconditionally.

## TIER 10 -- Conventions / slop

- **#12** 48 inline comment blocks of 8+ consecutive lines (worst:
  `parallelize.py:967` 33 lines; `lora.py:449` 29 lines with bug-history
  numbers). Triage all: one-line WHAT stays, WHY to module docstring or PR
  body.
- **#11** 8 of 56 recent commits carry `pytorch/torchtitan#N` / `Refs:`
  trailers (pre-rule history): PR slices must be fresh commits, never
  cherry-picks of these messages.
- **#4** Remaining internal references: `parallelize.py:1326`, `kcp.py:34`
  (probe script names), plus 2 HANDOFF/PLAN hits.
- **#5** `kcp.py` docstring's hand-rolled-halo history paragraph: compress to
  one sentence, move the story to the PR-D body.
- **#52** `test_vit_cp_plan.py:34` -- uneven-split video coverage dropped in
  the test rewrite; re-pin t>1 with blocks % group_size != 0.

## Core-footprint audit (for the PR slices)

- `tools/grouped_mm_empty_shim.py` (+42): zero importers -- delete from the
  fork (logbook copy exists); never on a PR branch. (#1 manual)
- `distributed/pipeline_parallel.py` (+8): post-#3856 microbatch-count fix --
  determine upstream-bug vs local-adaptation; PR candidate if the former. (#2)
- `models/utils.py` (+7): skip non-Shard placements -- generic fix, next small
  core PR candidate. (#3)
