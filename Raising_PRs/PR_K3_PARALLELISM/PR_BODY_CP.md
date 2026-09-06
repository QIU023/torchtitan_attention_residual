# PR title: [Kimi K3] Context parallelism for the text decoder: packed MLA kernels on the CP kernel stack, KCP on KDA

PR 4313. Branch `cp_review5` on the fork (`edc4cd71b`): stacked on the declarations PR (`k3_spmd_decl` `dbc60701d`, on upstream/main `390e2985b` with 4446 merged), then three commits. The first is the open upstream CP stack (PR 4322 / 4449 / 4450 at `860d5aa64d`) copied to unblock running, every copied file and class marked "copied from upstream open PR 4322/4449/4450 to unblock running; pending rebase and reconcile", the whole commit to be dropped at the rebase once the stack merges. The second (`4c2cd4d4c`) is this PR: the CP layer, re-stacked on 2026-09-05 with one conflict in `parallelize.py` against 4446 (the CP hook sits after the model assertion, before 4446's replicated annotation and mesh resolution). The third (`edc4cd71b`) moves the KDA kernel to Attention Gym's merged context-parallel recipe (attention-gym `b19162e`, 2026-09-04, or later) and drops the SM100/SM103 guard main's `kda.py` carried, since Attention Gym dispatches by capability itself now. The declarations the CP layer needs are the base PR's now; the previous head of this branch, `223e97a23`, carried its own copy of them, which is what the PR branch `k3_cp_text` holds until the sync is approved.

The PR composes with data parallelism and, optionally, expert parallelism; TP x CP is the TP PR's matter. The matrix below ran on `edc4cd71b` with attention-gym `b19162e` on 2026-09-05; every step-1 loss is bitwise the pre-merge recipe's, the later steps carry the merged kernels' rounding. The dp rows come with their expert-parallel twins and dp x cp at world 8.

--- PASTE BEGIN ---

### Summary

Stacked on 4492 (the declarations: the tower over cp and the multimodal inputs' layout are what CP needs under `spmd_types`; its diff shows here until it merges, and its TP entries are inert at tp = 1) and on the open CP kernel stack 4322 / 4449 / 4450, carried as copies until they merge. None of their files are this PR's.

Adds context parallelism to the Kimi K3 decoder on the CP kernel stack (PR 4322 / 4449 / 4450), with the model under `spmd_types` through its own declarations: every attention layer runs a `ContextParallelKernel` installed by a transform, and each kernel owns its collectives behind the identity boundary. Before this change `parallelize.py` rejects `context_parallel_degree > 1`.

After it the MLA layers run `MLAUlyssesCPFlexAttention` (the default) or `MLAAllGatherCPFlexAttention`, the generic Ulysses and all-gather kernels specialised to MLA's expanded key: MLA expands one rotary vector per token onto every head before the kernel, so the kernels split the key back, move the nope part packed with q and v (one all-to-all) or with v (one gather), move the rotary slice once as the headless vector it is, and expand it after the exchange -- the packed exchange of the first version of this PR, on the new interface.

The KDA layers run Attention Gym's context-parallel delta rule (`ContextParallelInnerKDA`, KCP: the sequence stays sharded end to end and the recurrence hands its state from rank to rank); `torchtitan_recipes.kimi_k3` applies the generic `ContextParallelTransform` to the MLA layers and a K3 transform to the KDA layers, which the generic transform and validation do not see because KDA is not an attention config.

### Copied from upstream, pending rebase

The first commit is the net diff of the open CP stack (4322 / 4449 / 4450, fegin) at `860d5aa64d`, so the kernels here can run on its interface before it lands: `models/common/cp_attention.py`, the `transforms` package, `distributed/context_parallel/validation.py`, the decoder's mask flag and the identity boundary in `decoder_sharding.py`, the trainer's validation call, and the stack's own moves of flux, qwen3, gpt_oss, muse_glimmer and the qwen3_5 recipes onto the transform, with its tests. Every copied file and class carries the note; the commit is dropped at the rebase once the stack merges, and the rest of this PR is the two commits below.

### Design

- The declarations (`sharding.py`, `model.py`, `parallelize.py`)
  - `spmd_types` consumes every declaration, so the dense path is declared at tp = 1 (weights whose inputs and consumers are TP-invariant are `I`, the rest `R`; the stream conversions at the model entry, the MLA entry, the MoE seams), and the MoonViT tower is invariant at TP and rank-local over cp with the cp axis on every vision layout.
  - `parallelize_kimi_k3` resolves the FSDP meshes through `resolve_fsdp_mesh` / `resolve_sparse_fsdp_mesh` and runs `model.parallelize` whenever `spmd_types` drives the model; every parameter is a DTensor on the full mesh, which FSDP under `spmd_types` requires, and the CP kernels take their group from the SPMD mesh.
- The MLA kernels (`context_parallel.py`)
  - `MLAUlyssesCPFlexAttention(UlyssesCPFlexAttention)`: `_split_rope` takes the nope part `[T, H, N]` and head 0's rope slice `[T, R]` off the expanded key; `(q | k_nope | v)` is one `_reshard` from `S(0)` to `S(1)`; the rope slice is one `spmd.redistribute` from `S(0)` to `R` (its backward a reduce-scatter in the activation dtype); the key is rebuilt on the local heads and `FlexAttention.forward` runs with the global mask (`shard_attention_mask = False` inherited); the output reshards back.
  - Against the generic kernel this saves the rope slice's $H - 1$ copies per token and folds three exchanges into one.
  - `MLAAllGatherCPFlexAttention(AllGatherCPFlexAttention)`: `(k_nope | v)` and the rope slice are gathered from `S(0)` to `R` with the inherited `reduce_dtype`; q and the mask stay token-sharded. Against the generic kernel this gathers $H \cdot R - R$ fewer values per token.
  - Both carry `rope_head_dim` in their config; the recipe fills it from the attention config's `qk_rope_head_dim` through `kernel_config_overrides`, and a generic kernel takes the expanded key as is. The split is the cost fegin's review accepted for the unified `(q, k, v)` interface: one copy of the packed tensor per layer, and the expanded key already exists on the way in.
- KDA (`context_parallel.py`, `kda.py`): `ContextParallelInnerKDA` runs Attention Gym's context-parallel delta rule (`context_parallel_kda`, with `context_parallel_conv_history` for the conv's history); the kernel and its numerics are Attention Gym's, this branch only builds the plan (`ContextParallelPlan.from_fragments`: one document, equal contiguous shards) and its routing (`plan.routing`, device tensors sized by the span and the conv's history, one per shape), and splits `InnerKDA` into `_pack_inputs` and `_conv_and_scan` so the CP kernel adds only the history and the routing. Packed-document boundaries under KCP raise `NotImplementedError` for now.
- The transforms (`torchtitan_recipes/kimi_k3.py`): `KimiK3DeltaContextParallelTransform` retypes every `inner_kda` to the KCP kernel; `kimi_k3_context_parallel(config, cp_degree=..., mla_kernel=...)` sets the degree, turns the load balancer off (both kernels read the sequence as rank-ordered contiguous chunks), selects `spmd_types`, and applies the generic transform for MLA and the K3 one for KDA; the cp2 flavors are that call. The model checks at config time that every KDA layer got its kernel, since upstream validation covers the attention layers only.
- The boundary and the model: `set_gqa_inner_attention_local_map` (the stack's identity boundary) on the MLA inner attention; the head count is derived from the projection width so TP and CP compose without a branch; `apply_cp_kimi_k3` hands the model its cp group for the vision splice, whose plain-tensor branch rejects sequence parallel like the DTensor branch.

### Results

`kimi_k3_debugmodel` (the multimodal debug model, whose vision splice under CP is the one piece of the model body the kernels do not cover; the flavor pins `partial_dtensor`, so the dp cells pass `spmd_types` and the cp flavors select it).

`--debug.seed 42 --debug.deterministic`, one seed checkpoint shared by every flavor, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read (FlexAttention's autotune moves this model's later steps between compile-cache states; step 1 does not move). The runner with the seed-load assertion is `phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh` in the logbook.

Measured on an RTX 5060 Ti (SM120), where Attention Gym routes KDA through its portable kernels; the branch carries no local patch for it (the SM100/SM103 guard in `kda.py` is gone). The generic rows run the upstream kernels through the same recipe, so the packed kernels are read against them on the same seed and batch.

```
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable"
torchrun --nproc_per_node=1 $COMMON --config kimi_k3_debugmodel --training.steps 1 --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
S="--parallelism.spmd_backend spmd_types"; D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"; E="--parallelism.expert_parallel_degree"
cell dp1 1 --config kimi_k3_debugmodel $D 1 $S;  cell dp2 2 --config kimi_k3_debugmodel $D 2 $S
cell cp2 2 --config kimi_k3_debugmodel_cp2 $D 1;  cell cp2_ag 2 --config kimi_k3_debugmodel_cp2_allgather $D 1
cell cp4 4 --config kimi_k3_debugmodel_cp2 $D 1 $C 4;  cell cp8 8 --config kimi_k3_debugmodel_cp2 $D 1 $C 8;  cell dp2_cp2 4 --config kimi_k3_debugmodel_cp2 $D 2;  cell dp2_ep2_cp2 4 --config kimi_k3_debugmodel_cp2 $D 2 $E 2
```

Every cell ran twice on the same seed checkpoint and the second run is read. cp8 starts 1e-2 above cp2 and cp4 at step 1 (12.54963 against 12.53972 and 12.53932, reproduced on both of its runs); the same cell on upstream's generic Ulysses kernel reads the same 12.54963, and at step 1 the packed kernel's per-parameter gradients at cp8 sit at the same distance from dp1 as cp2's (median 1.2e-2 against 1.1e-2, max 3.3e-1 against 4.6e-1, the tail on 16-element norm weights and `A_log`) and within bf16 rounding of the generic kernel at cp8 (median 1.8e-4, 24 of 750 parameters identical), so the offset belongs to the CP degree, not to the packing.

| cell | world | MLA kernel | KDA | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | 1 | - | - | 12.52977 | 7.36833 | 2.91045 |
| dp2 | 2 | - | - | 12.53137 | 7.25082 | 3.15411 |
| dp2 x ep2 | 2 | - | - | 12.53146 | 7.13441 | 3.09174 |
| cp2 | 2 | packed Ulysses (this PR) | KCP | 12.53972 | 7.20787 | 2.98935 |
| cp2 | 2 | generic Ulysses (4450) | KCP | 12.53972 | 7.27002 | 3.01910 |
| cp2 | 2 | packed all-gather KV (this PR) | KCP | 12.53972 | 7.31401 | 3.04420 |
| cp2 | 2 | generic all-gather KV (4322) | KCP | 12.53972 | 7.33010 | 2.92453 |
| cp4 | 4 | packed Ulysses | KCP | 12.53932 | 7.11244 | 3.09557 |
| cp8 | 8 | packed Ulysses | KCP | 12.54963 | 7.26635 | 3.00692 |
| cp8 | 8 | generic Ulysses (4450) | KCP | 12.54963 | 7.33906 | 3.08600 |
| dp2 x cp2 | 4 | packed Ulysses | KCP | 12.52908 | 7.27039 | 3.15622 |
| dp2 x ep2 x cp2 | 4 | packed Ulysses | KCP | 12.52759 | 7.23991 | 3.11484 |
| dp2 x cp4 | 8 | packed Ulysses | KCP | 12.53067 | 7.12889 | 3.16472 |
| dp2 x ep2 x cp4 | 8 | packed Ulysses | KCP | 12.52663 | 7.12298 | 3.10091 |
| dp4 x cp2 | 8 | packed Ulysses | KCP | 12.54269 | 6.95012 | 3.10984 |
| dp4 x ep2 x cp2 | 8 | packed Ulysses | KCP | 12.53850 | 6.93310 | 2.97357 |
| dp4 x ep4 x cp2 | 8 | packed Ulysses | KCP | 12.53869 | 6.94088 | 3.09810 |

The four MLA kernels agree at step 1 to the digit and part afterwards: a packed kernel moves the same values as its generic counterpart but sums the rope slice's gradient in a different order (over the local heads first, then the reduce-scatter across cp), one bf16 rounding of a sum, the same class of difference as the two transports in the pipeline PR.

Step-1 per-parameter gradients (fp32 norm of every parameter's gradient, hashed; rank 0's local gradient, 750 parameters, one shared seed, each kernel on its own warm compile cache; measured with the same kernels on the earlier cut of this branch) are the evidence for what a packed kernel changes against its generic counterpart:

| comparison (step 1, cp2; measured on the pre-merge recipe, whose forward is the same function, as the step-1 losses above show) | loss | sha1-identical parameters | relative difference of the per-parameter norm: median / p90 / max |
|---|---|---|---|
| packed Ulysses vs generic Ulysses (4450) | identical, 12.53972 | 24 of 750 | 1.6e-4 / 1.5e-3 / 2.1e-2 |
| packed all-gather vs generic all-gather (4322) | identical | 24 of 750 | 1.1e-4 / 1.0e-3 / 8.8e-3 |
| packed Ulysses vs packed all-gather | identical | 22 of 750 | 1.8e-4 / 1.5e-3 / 1.5e-2 |
| packed Ulysses vs generic Ulysses (4450) at cp8 | identical, 12.54963 | 24 of 750 | 1.8e-4 / 1.3e-3 / 3.1e-2 |

The maxima sit on 16-element `A_log` vectors and residual norms whose gradient norm is 1e-4: the distribution bf16 summation order produces, with no parameter group standing out.

Against a single GPU the reference is not bitwise on this model: gradients are kept in bf16, so any re-partitioning of the arithmetic moves every parameter by about one bf16 rounding. The cp-reduced full gradient of each CP cell is compared with dp1 below, next to plain data parallelism on the same tree, seed and batch; the four CP kernels give the same numbers to two digits, and CP sits below the control.

| comparison (step 1, cp-reduced full gradient, 750 parameters) | loss | relative difference of the per-parameter norm: median / p90 / max |
|---|---|---|
| dp1 vs dp2 (data parallel; the loader shards documents by rank, so the batch composition changes too: an upper bound) | 12.52977 vs 12.53137 | 2.5e-2 / 7.9e-2 / 5.2e-1 |
| dp1 vs cp2, packed Ulysses | 12.52977 vs 12.53972 | 1.1e-2 / 6.1e-2 / 4.6e-1 |
| dp1 vs cp2, generic Ulysses / packed all-gather / generic all-gather | same | 1.1e-2 / 6.1e-2 / 4.6e-1 each |
| dp1 vs cp8, packed Ulysses | 12.52977 vs 12.54963 | 1.2e-2 / 6.3e-2 / 3.3e-1 |
| dp1 vs cp8, generic Ulysses (4450) | same | 1.2e-2 / 6.1e-2 / 3.3e-1 |
| cp2 vs cp8, packed Ulysses | 12.53972 vs 12.54963 | 9.5e-3 / 6.4e-2 / 3.7e-1 |

The maxima are again the 16-element `A_log` vectors with gradient norms below 1e-4; within a cell the two cp ranks hold identical reduced gradients (750 of 750 sha1-equal).

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py                           +312/-0  the declarations: the dense path at tp = 1, the tower over cp, the MoE seams
      model.py                              +261/-14  the declaration calls and local regions; the KDA-kernel check; the cp group and the vision splice under CP
      parallelize.py                        +44/-19  the spmd_types backend branch; context parallel off the unsupported list; apply_cp_kimi_k3
      context_parallel.py                   +273/-0  the packed MLA kernels, the KCP kernel, the plan and its routing (new)
      kda.py                                +70/-12  InnerKDA split into pack / conv-and-scan; the KCP branch in the kernel; the capability guard removed; head views with -1
    torchtitan_recipes/
      kimi_k3.py                            +85/-0   the KDA transform and the recipe helper (new)
      tests/features.py                     +24/-0   the cp2 and cp2 all-gather configurations
    tests/unit_tests/cpu/
      test_kimi_k3_cp_kernels.py            +208/-0  the packed kernels' exchanges and their round trip, the transforms, the KDA check (new)
    tests/integration_tests/features.py     +8/-0    the cp2 cell

### CI/CD Coverage

Seven CPU unit tests in the default suite (the packed kernels move exactly the packed tensor and the rope slice and hand FlexAttention what MLA produced; the transforms install a kernel on every layer and keep non-default fields; the KDA check); a cp2 integration cell on two GPUs (skipped on ROCm). The all-gather flavor is in the recipes for the run above and is not a CI cell.

### Review round 1

- Ulysses in the shape of 3978 / 4322 (tianyu-l), and "keep yours but use the new interface" (fegin): done as option (a) of fegin's comment. The packed exchange lives inside `MLAUlyssesCPFlexAttention`, a subclass of 4450's kernel; the inner attention keeps the `(q, k, v)` interface and the kernel splits the expanded key before packing. If the copy ever shows in a profile, the `MLAAttention` interface that hands `k_nope` and `k_rope` over separately is the next step.
- All-gather KV for MLA (tianyu-l, fegin): the generic kernel from 4322 works on this model as is; `MLAAllGatherCPFlexAttention` is the extend version that gathers the packed nope key and v and the rope slice once. Both are in the table above against their generic counterparts.
- The Attention Gym version of KCP (tianyu-l): the KDA layers run Attention Gym's `context_parallel_kda`; the kernel is Attention Gym's to own, this branch only wires it.
- Document boundaries under CP (drisspg): the Ulysses kernels keep the mask global (`shard_attention_mask = False`), so every rank attends with the same causal x document mask as the non-CP path after the exchange; the all-gather kernels take the mask sharded along q like every other model; KCP runs one document per batch and refuses packed boundaries explicitly instead of scanning across them.
- The text-only variant question on the vision splice (tianyu-l): the debug model stays the multimodal one, as the expert-parallel review settled, so a text-only variant is not added here; the dependency this PR keeps is the CP-shard case, a shard whose positions hold no image tokens still has to issue the tower's collectives, inlined in the splice.
- ROCm: the cp2 cell is skipped there.

--- PASTE END ---
