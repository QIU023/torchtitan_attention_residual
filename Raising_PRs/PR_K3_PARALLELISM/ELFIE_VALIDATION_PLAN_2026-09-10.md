# Kimi K3 on the new tree: what to run on GB200 (2026-09-10)

For Elfie (PR 4281's reviewer and the only one of us with multi-node GB200/GB300 time). Everything below is on branches of `QIU023/torchtitan` based on upstream main; the old tree that PR 4281 pointed at is frozen and nothing on it is worth another run. Two parts: (1) what only you can measure, in the order it unblocks PRs; (2) the cells of your 08-26/27 old-tree table, re-mapped to the new tree, with what we already hold on one node so you know what "pass" looks like.

## 0. Setup

- Environment: torch nightly `2.15.0.dev20260906` or later (cu128 or cu130), Attention Gym at `QIU023/attention-gym` `b19162e` on `PYTHONPATH` (or `meta-pytorch/attention-gym` main), `spmd_types==0.2.5`, `torch_remat` as pinned in `pyproject.toml`. GB200 is SM100: Attention Gym selects its CuTe KDA kernels there, while our boxes (SM120, A100) and the H100 in PR 4500 run the portable Triton ones, so numbers from you and from us are not expected to agree bitwise; what is expected bitwise is stated per item.
- Branches (heads as of 2026-09-10):
  - `k3_on_4025` = `da7f9e348` on main `ac10ca48f`: the full integration tree (every feature below at once). Use it for the combined cells.
  - `k3_pp_text` = `75045fed5` (PR 4312, text pipeline + your transport fix ported), `k3_pp_mm` = `c87097ae5` (PR 4381, DEP on top), `k3_cp_text` = `61a73ca6c`, `k3_cp_mm` = `a063a3d0e` (PR 4380), `k3_tp_sp` = `9a62f5229` (PR 4499), `k3_qb` = `895f4d6e9` (PR 4412). Use these for the per-PR cells.
- Protocol we use everywhere: `--debug.seed 42 --debug.deterministic --metrics.log_freq 1`, the 33-layer `kimi_k3_debugmodel` (main's), 4096 tokens per step in 256-token micro-batches unless stated, spmd_types (`--parallelism.spmd_backend spmd_types`, the default). Report loss and grad norm at steps 1, 3, 10 per cell; "bitwise" means identical printed values.

## 0b. Is the tree current? (checked 2026-09-10 evening)

`k3_on_4025` = `k3_int_20260910` = `da7f9e348` bundles every feature, but it is behind five of today's branch heads; nothing in the list changes a number the plan asks for, and the two-node PP ask (1.1) is on `k3_pp_text`, which is current.

| feature | tree carries | branch head today | gap |
| --- | --- | --- | --- |
| TP/SP | the pre-rework stack (core `clip_grad_norm_` grouping, no partial_dtensor refusal, no b200 cell) | `k3_tp_sp` `9a62f5229` | the rework was bitwise for spmd_types cells; the tree still edits core `distributed/utils.py` |
| LoRA | 8 commits | `lora_review2` `72bbcb639` | the user's 5 commits of 2026-09-10 (typing, test location, torchao 0.18 NF4 import, flavor cleanup, flake8) |
| quantile balancing | our #4412 | maintainers' #4577 supersedes it | rebase onto Shuhua's router once it merges |
| Kimi-Linear graft | absent | `k3_linear_graft` `3b6f03347` | new today, fork-internal |
| empty container / QAT / MTP / AC | the ports | the standalone branches add tests and typing on top | tests only |

So: run the plan on the tree as it is; the tree gets re-bundled after #4577 and the LoRA commits, and that re-bundle is the one to hand over for the full-scale cells (1.3).

## 1. What only you can measure, in the order it unblocks PRs

### 1.1 Two-node PP8 with the ported transport fix (PR 4312)

The hang you found on two GB200 nodes under PP=8 and 1F1B. Branch `k3_pp_text` (`75045fed5`) carries two commits on top of the stage design: an eager edge-communicator warm-up (default) and your fix as an opt-in, `TORCHTITAN_PIPELINE_NEIGHBOR_P2P=1` (one two-rank NCCL group per adjacent stage edge, metadata over a per-replica Gloo group, the inference-mode vote as one all-reduce; composed as a mixin in front of `AttnResPipelineStage`). Background: `PP_TRANSPORT_NOTE_FOR_ELFIE.md`, and one more finding since: with the device bound at init, no communicator is created lazily during the schedule on this torch, so the warm-up is a no-op by construction and only the isolation can matter.

Run, 2 nodes x 4 GPUs, three times:

```
# plain 1F1B, eight micro-batches
torchrun --nnodes 2 --nproc_per_node 4 ... -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 10 \
  --training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 8 \
  --parallelism.pipeline_parallel_schedule 1F1B --parallelism.num-pp-microbatches 8
# the same with TORCHTITAN_PIPELINE_NEIGHBOR_P2P=1 in the environment
# the pp8 x vp4 recipe (Interleaved1F1B): --module torchtitan_recipes.tests.features --config kimi_k3_debugmodel_pp8_vp4
```

Report: hang or not (with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` captured for the first minute of a hang), and the step-1 loss against one node. Expected: step 1 identical between the switch on and off (the fix moves no tensor value), and identical to a one-node run of the same command on your machine. On one node we hold pp2 / pp8 1F1B / pp8 x vp4 bitwise with the switch on and off.

### 1.2 Optimizer state on resume

Your second finding on PR 4281 (Adam state missing for a layer-0 parameter). Main's `dc3985ad4` (#4474) materialises optimizer state for every trainable parameter before save and load, and on the new tree layer 0 has no attention-residual parameter at all. A round trip on our box is exact for weights, Adam moments, step and learning rate. Worth one check from you at your scale: save at step 5, resume, compare step 6 to an uninterrupted run. Expected: the checkpoint contents identical; step 6 within the run-to-run class of your machine (we see a 1e-3 difference at the resumed step's backward that is runtime determinism, not checkpoint semantics; a second resume reproduces the first bitwise).

### 1.3 Official weights, full size: the step-0 anchor, then N steps

The debug flavor cannot anchor correctness; the released weights can. Two steps:

1. Step-0 anchor: load the released Kimi K3 checkpoint into the `Kimi-K3` flavor (the official topology field for field) through the adapter, which takes the MXFP4-packed experts through torch's quantized HuggingFace reader (`--checkpoint.initial_load_in_hf` plus the quantized-load flag on the branch), and compare the step-0 loss on a fixed batch with a vLLM or HF forward on the same batch. Expected: equal to bf16 rounding of the logits. Our released-format checks so far are on the debug-size released-layout artifact only (`report_arch`, 0 missing / 0 extra keys, bitwise export round trip).
2. Then FSDP x TP x EP (x PP) at the released shape for N steps, the combination you run in production; loss finite and decreasing is the bar, plus per-rank peak memory.

### 1.4 MoonEP

Needs NVSwitch multicast; none of our machines has it. Branch `k3_on_4025`, `model_registry(..., moe_comm_backend="moonep")` with the `moonep` package installed. The dispatcher and the expert side (`moon_ep_experts.py`: the [E+B] weight tables, prefetch, slot-gradient reduce) are exercised end to end only through an in-process fake on CPU; on hardware the table allocation over `moonep.buffer`'s VMM primitives (`MoonEPTableBackendNVLink`) raises until it is written against the installed package, so the first run is a porting task, not a measurement. Compare ep2 x fsdp2 for three steps against the standard dispatcher: same step-1 loss, gradients within the backend's own arithmetic class.

### 1.5 DEP hiding rate at a size that can show it

`k3_pp_mm` (`c87097ae5`): `kimi_k3_debugmodel_pp8_vp4_vit_dep` (the tower and the embedding on the first stage) with `vit_bubble` or `vit_prefetch`. At debug size the tower is too cheap to hide anything; run with at least 32 micro-batches and a real tower cost ratio (the released 27-layer tower against the text stages), decide the criteria beforehand (backward hidden >= 90%, forward >= 25%), read them off the bubble plan's log lines ("planned encodes in bubbles / synchronous / idle slots"). Numerics: step 1 bitwise with the same recipe without `vit_dep`.

### 1.6 Dynamic vision CP on real images and video

`k3_cp_mm` (`a063a3d0e`): with `context_parallel_degree > 1` every image at or above 256 patches is split over a sub-CP group (log line "Dynamic CP: N large image(s) ... over M sub-CP group(s)"). Debug data has one image per step; what needs your data is the load-balancing case (several large images per step, `balance_images`, sub-groups of 2 and 4) and video (bands are the same rows of every frame). Numerics: against the replicated tower we hold fp32 agreement to 1e-6 (forward) and 1e-5 (gradients) on one image, a padded image, a two-frame video and a mixed batch. Also the CP x TP combination once PR 4500 is in.

### 1.7 pp_balance and attention-residual offload at real width

`k3_on_4025`: `attn_res_cache_offload` (the rank store on pinned host memory) and `pp_balance` (saved activations parked on a peer through the Mooncake Transfer Engine). At debug width rank 3's peak is dominated by the logits, so the activation-dominated profile they target is invisible; report per-rank peak memory with and without each on the released shape, and the RDMA path of Mooncake.

### 1.8 QAT accuracy

Your own suggestion. `kimi_k3_debugmodel_mx_qat` (MXFP4 weights / MXFP8 activations, fake-quant on the routed experts, scope = every `GroupedExperts`, which is exactly what the released index quantises). Our offline attribution on the debug model: 0.47 nat from the weights, 0.24 nat from the activations at step 1; a study at size is yours.

### 1.9 The 100-step tp4 / tp8 rows

`k3_tp_sp` (`9a62f5229`): the A100 kit (`phase13_k3like_48b_posttrain/matrix_scripts/tp_a100/run_bf16_100.sh`) gives tp1 / tp2 / tp4 at 100 steps in PR 4500's format; tp8 needs eight GPUs of one node and is the row we do not have.

## 2. Your 08-26/27 old-tree cells, re-mapped

| your cell (old tree) | new-tree equivalent | branch | one-node result we hold |
| --- | --- | --- | --- |
| 1 GPU: KDA, MLA, routed MoE, AttnRes finite loss | `kimi_k3_debugmodel` dp1 | any; `k3_on_4025` | 12.40087 / 10.55432 / 7.71281 at steps 1-3, bitwise across pp splits |
| 4 GPU: TP2 x CP2 | `--parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2 --parallelism.no-enable-sequence-parallel` with the cp2 recipe | `k3_on_4025` | 18-cell matrix cell `tp2_pp2_cp2` and `fsdp2_tp2_cp2` seed-ok (12.39383 / 12.39093 at step 1) |
| 4 GPU: FSDP2 x TP2 x EP2 | `--parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2` | `k3_on_4025` or `k3_tp_sp` | `ep2_fsdp2_tp2_cp2` 12.38950 / 7.49985; dp2 x ep2 x tp2 in the TP body's dp2 stream |
| 2 nodes: PP4 | `--parallelism.pipeline_parallel_degree 4` (1F1B) or the pp8 x vp4 recipe | `k3_pp_text` | pp4 12.40087 (bitwise with dp1) / 7.82246; see 1.1 for the two-node ask |
| 4 nodes: FSDP2 x TP2 x CP2 x PP2 + EP2 | the same flags on `k3_on_4025` | `k3_on_4025` | one node holds `ep2_fsdp2_pp2_cp2` 12.38757 / 7.66995 and `ep2_fsdp2_tp2_pp2` 12.38958 / 7.44238 (8 GPUs); the 4-node combination is yours |
| MTP (old `mtp_loss.py`, the `positions=` failure) | `kimi_k3_debugmodel_mtp` (the `mtp.py` layer; refuses SP and CP) | `k3_on_4025` | dp1 runs; main's own MTP + CP (#4544) is not in this tree yet |
| the NCCL hang | 1.1 above | `k3_pp_text` | |
| the optimizer-state resume | 1.2 above | any | |

What to send back per cell: the command, the environment (torch, Attention Gym commit, NCCL version), steps 1 / 3 / 10 loss and grad norm on rank 0, peak memory per rank, and for anything that hangs the NCCL INIT/NET log of the first minute.
