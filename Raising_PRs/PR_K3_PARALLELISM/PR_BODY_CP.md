# PR title: [Kimi K3] Context parallelism for the text decoder: packed MLA kernels on the CP kernel stack, KCP on KDA

PR 4313. Branch `cp_review4` on the fork (`624dd6408`). It sits on `cp_base_stack` (`a8a6f331f`), a scratch commit that is fegin's CP stack (PR 4322 / 4449 / 4450 at `860d5aa64d`) applied onto upstream/main `9b5f60c40`; above it the TP/SP commits (`tp_review2`), the spmd declarations (`spmd_review2`), the CP content of `cp_review3` (`a4322344d`) and the commit that moves it onto the stack (`624dd6408`). The PR branch is synced only on the user's approval, and only once the stack has landed (the scratch commit is never filed). Paste between the markers; the header of the PR should say it stacks on the TP/SP and declaration commits and on fegin's CP stack.

--- PASTE BEGIN ---

### Summary

Adds context parallelism to the Kimi K3 text decoder on the CP kernel stack (PR 4322 / 4449 / 4450): every attention layer runs a `ContextParallelKernel` installed by a transform, and each kernel owns its collectives behind the identity boundary. Before this change `parallelize.py` rejects `context_parallel_degree > 1`. After it the MLA layers run `MLAUlyssesCPFlexAttention` (the default) or `MLAAllGatherCPFlexAttention`, the generic Ulysses and all-gather kernels specialised to MLA's expanded key: MLA expands one rotary vector per token onto every head before the kernel, so the kernels split the key back, move the nope part packed with q and v (one all-to-all) or with v (one gather), move the rotary slice once as the headless vector it is, and expand it after the exchange -- the packed exchange of the first version of this PR, on the new interface. The KDA layers run Attention Gym's context-parallel delta rule (`ContextParallelInnerKDA`, KCP: the sequence stays sharded end to end and the recurrence hands its state from rank to rank). `torchtitan_recipes.kimi_k3` applies the generic `ContextParallelTransform` to the MLA layers and a K3 transform to the KDA layers, which the generic transform and validation do not see because KDA is not an attention config.

### Design

- The MLA kernels (`context_parallel.py`)
  - `MLAUlyssesCPFlexAttention(UlyssesCPFlexAttention)`: `_split_rope` takes the nope part `[T, H, N]` and head 0's rope slice `[T, R]` off the expanded key; `(q | k_nope | v)` is one `_reshard` from `S(0)` to `S(1)`; the rope slice is one `spmd.redistribute` from `S(0)` to `R` (its backward a reduce-scatter in the activation dtype); the key is rebuilt on the local heads and `FlexAttention.forward` runs with the global mask (`shard_attention_mask = False` inherited); the output reshards back. Against the generic kernel this saves the rope slice's $H - 1$ copies per token and folds three exchanges into one.
  - `MLAAllGatherCPFlexAttention(AllGatherCPFlexAttention)`: `(k_nope | v)` and the rope slice are gathered from `S(0)` to `R` with the inherited `reduce_dtype`; q and the mask stay token-sharded. Against the generic kernel this gathers $H \cdot R - R$ fewer values per token.
  - Both carry `rope_head_dim` in their config; the recipe fills it from the attention config's `qk_rope_head_dim` through `kernel_config_overrides`, and a generic kernel takes the expanded key as is. The split is the cost fegin's review accepted for the unified `(q, k, v)` interface: one copy of the packed tensor per layer, and the expanded key already exists on the way in.
- The KDA kernel (`context_parallel.py`, `kda.py`): `ContextParallelInnerKDA` builds the routing plan (`kcp_plan`: one document, equal contiguous shards, the sharding the trainer applied), takes the previous rank's conv tail as history (`context_parallel_conv_history`), and runs `context_parallel_kda`, which exchanges per-fragment affine state summaries so each rank scans from its true entry state; `InnerKDA` is split into `_pack_inputs` and `_conv_and_scan(..., conv_state, cp_plan, cp_group)` so the CP kernel adds only the history and the plan. Packed-document boundaries under KCP raise `NotImplementedError` for now.
- The transforms (`torchtitan_recipes/kimi_k3.py`): `KimiK3DeltaContextParallelTransform` retypes every `inner_kda` to the KCP kernel; `kimi_k3_context_parallel(config, cp_degree=..., mla_kernel=...)` sets the degree, turns the load balancer off (both kernels read the sequence as rank-ordered contiguous chunks), selects `spmd_types`, and applies the generic transform for MLA and the K3 one for KDA; the cp2 flavors are that call. The model checks at config time that every KDA layer got its kernel, since upstream validation covers the attention layers only.
- The boundary and the model: `set_gqa_inner_attention_local_map` (the stack's identity boundary) on the MLA inner attention; the head count is derived from the projection width so TP and CP compose without a branch; `apply_cp_kimi_k3` hands the model its cp group for the vision splice, whose plain-tensor branch rejects sequence parallel like the DTensor branch.

### Results

`kimi_k3_debugmodel`, `--debug.seed 42 --debug.deterministic`, one seed checkpoint per flavor, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read (FlexAttention's autotune moves this model's later steps between compile-cache states; step 1 does not move). The runner with the seed-load assertion is `phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh` in the logbook. Measured on an RTX 5060 Ti (SM120) with Attention Gym's SM100/SM103 guard on the KDA kernel lifted locally, which routes it through Attention Gym's portable kernels; that patch is not on the branch. The generic rows run the upstream kernels through the same recipe, so the packed kernels are read against them on the same seed and batch.

```
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --parallelism.data_parallel_shard_degree 1"
torchrun --nproc_per_node=1 $COMMON --config kimi_k3_debugmodel --training.steps 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
S="--parallelism.spmd_backend spmd_types"; T="--parallelism.tensor_parallel_degree 2"
cell dp1 1 --config kimi_k3_debugmodel $S;  cell tp2 2 --config kimi_k3_debugmodel $T $S
cell cp2 2 --config kimi_k3_debugmodel_cp2;  cell cp2_ag 2 --config kimi_k3_debugmodel_cp2_allgather
cell tp2cp2 4 --config kimi_k3_debugmodel_cp2 $T --parallelism.no-enable-sequence-parallel
```

<!-- TBD: fill from /workspace/mx3_cp4_* -->
| cell | world | MLA kernel | KDA | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | 1 | - | - | | | |
| tp2 | 2 | - | - | | | |
| cp2 | 2 | packed Ulysses (this PR) | KCP | | | |
| cp2 | 2 | generic Ulysses (4450) | KCP | | | |
| cp2 | 2 | packed all-gather KV (this PR) | KCP | | | |
| cp2 | 2 | generic all-gather KV (4322) | KCP | | | |
| tp2 x cp2 (no SP) | 4 | packed Ulysses | KCP | | | |

### Changed files

    torchtitan/models/kimi_k3/
      context_parallel.py                   +246/-0  the packed MLA kernels, the KCP kernel, the plan (new)
      kda.py                                +69/-5   InnerKDA split into pack / conv-and-scan; the KCP branch in the kernel
      model.py                              +115/-3  the KDA-kernel check, the cp group for the vision splice, the head count from the projection width
      sharding.py                           +8/-4    the identity boundary on the MLA inner attention
      parallelize.py                        +22/-5   context parallel off the unsupported list; apply_cp_kimi_k3
    torchtitan_recipes/
      kimi_k3.py                            +85/-0   the KDA transform and the recipe helper (new)
      tests/features.py                     +24/-0   the cp2 and cp2 all-gather flavors
    tests/unit_tests/cpu/
      test_kimi_k3_cp_kernels.py            +208/-0  the packed kernels' exchanges and their round trip, the transforms, the KDA check (new)
    tests/integration_tests/features.py     +8/-0    the cp2 cell

### CI/CD Coverage

Seven CPU unit tests in the default suite (the packed kernels move exactly the packed tensor and the rope slice and hand FlexAttention what MLA produced; the transforms install a kernel on every layer and keep non-default fields; the KDA check); a cp2 integration cell on two GPUs (skipped on ROCm). The all-gather flavor is in the recipes for the run above and is not a CI cell.

### Review round 1

- Ulysses in the shape of 3978 / 4322 (tianyu-l), and "keep yours but use the new interface" (fegin): done as option (a) of fegin's comment. The packed exchange lives inside `MLAUlyssesCPFlexAttention`, a subclass of 4450's kernel; the inner attention keeps the `(q, k, v)` interface and the kernel splits the expanded key before packing. If the copy ever shows in a profile, the `MLAAttention` interface that hands `k_nope` and `k_rope` over separately is the next step.
- All-gather KV for MLA (tianyu-l, fegin): the generic kernel from 4322 works on this model as is; `MLAAllGatherCPFlexAttention` is the extend version that gathers the packed nope key and v and the rope slice once. Both are in the table above against their generic counterparts.
- The Attention Gym version of KCP (tianyu-l): the KDA layers run `attn_gym.linear.kda.context_parallel_kda` with `attn_gym.linear.context_parallel.context_parallel_conv_history`; fla is gone from this path.
- Document boundaries under CP (drisspg): the Ulysses kernels keep the mask global (`shard_attention_mask = False`), so every rank attends with the same causal x document mask as the non-CP path after the exchange; the all-gather kernels take the mask sharded along q like every other model; KCP runs one document per batch and refuses packed boundaries explicitly instead of scanning across them.
- The text-only variant question on the vision splice (tianyu-l): open for the maintainers to decide; the splice today keeps one model, inlines its zero-valued dependency on the tower's output so every rank issues the tower's collectives, and its plain-tensor branch rejects sequence parallel like the DTensor branch.
- ROCm: the cp2 cell is skipped there.

--- PASTE END ---
