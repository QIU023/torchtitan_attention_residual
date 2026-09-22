# PR title: [Kimi K3] MoonEP as a MoE comm backend, on the standard dispatcher seam

Branch `k3_moonep_seam` = `dce0f9604`, six commits on main `6c2dadbb3` (2026-09-18); 13 files, +1026/-12, of which 370 lines are tests. Open as draft PR 4751, whose head follows this branch. Earlier heads, for the notes below: `9e71d1073` (rebase onto `a3a819c67`, 2026-09-17), `904473aef` (the fused transport), `3c458bdf1` (lint), `84f2704ce` (the on-device test). Draft until the package is public on the CI boxes; the CPU tests need neither the package nor a GPU. Body in the #4577 format (2026-09-14), Results from the 4 x H100 box (`phase13_k3like_48b_posttrain/MOONEP_H100_2026-09-19.md`); cells in `MOONEP_TEST_PLAN_2026-09-16.md`.

Notes for filing:
- 2026-09-22: refactored to where torchtitan keeps transports, and the history rewritten to six typed commits, code before tests: the core capacity declaration, the transport primitives, the dispatcher with the experts, the Kimi K3 selection, the CPU tests, the on-device test with the h100 cell. Every commit parses and imports on its own tree; the final tree is unchanged by the split (earlier heads `7c4041c07`, `e09f8a5e0`). The primitives are `distributed/moonep/moonep.py`, the dispatcher sits with DeepEP and HybridEP in `models/common/token_dispatcher.py`, the prefetch-slot experts next to `GroupedExperts` in `models/common/moe.py`, and `"moonep"` is selected through `make_token_dispatcher_config`; the Kimi K3 folder keeps 49 lines of wiring. One `requires_ep` class variable replaces the two flags, the SM budget is a module constant rather than a config knob, and the barrier handles MoonEP does not expose are read in one named function. `kimi_k3_debugmodel_moonep` and the h100 cell `kimi_k3_moonep_fsdp4_ep4` follow `qwen3_moe_deepep` and its cell; without them nothing in the repo reaches the backend. CPU 22 passed.
- 2026-09-22, CORRECTION to the DeepEP note below: DeepEP v2 does not need an RDMA NIC. `torchtitan/models/qwen3/config_registry.py` documents the NVLink-only path (`EP_DISABLE_GIN=1`, `EP_REUSE_NCCL_COMM=0`, `NVSHMEM_REMOTE_TRANSPORT=none`, `NVSHMEM_DISABLE_MNNVL=1`) and torchtitan's own `run_8xgpu_integration_tests.sh` and `validate_release_gpu.sh` launch with `NCCL_NVLS_ENABLE=0 EP_DISABLE_GIN=1`. That text was in the tree at this branch's base, so the 09-19 run should have tried it before concluding the box could not host DeepEP. The comparison the report makes is therefore runnable on the same 4 x H100 NVSwitch box, and it is the missing evidence this PR needs.
- 2026-09-22, on the step-time table: those numbers come from the numerics cells, which run `--debug.deterministic` without compile at 256 tokens per micro-batch per rank and eight gradient-accumulation micro-batches. A step there is 5.2 seconds, so the total is set by the deterministic reference paths rather than by the MoE transport, and a difference of 83 ms between two such totals cannot be attributed to the transport. The shape is also structurally against MoonEP: it moves expert rows so tokens need not move, and at this flavor the rows it copies each micro-batch are an order of magnitude more bytes than the tokens they replace. A perf claim needs its own cells with compile on, determinism off and a per-rank token count near the target.
- 2026-09-19, 4 x H100 box: four commits, `822642a22` (the mesh check reads core's `efsdp` axis), `25dae14b4` (the CPU tests drop what the on-device test now covers), `676f91826` (ufmt) and `c02e6240f` (the mesh precondition tested against real `ParallelDims` meshes). An earlier version of this note said they were held back from the published branch, which was true when it was written: they went out afterwards on an explicit instruction, and because the push rewrote the branch it was preceded by a file-by-file check that the remote's previous eight commits were content-equivalent to the rebased branch's first eight. Draft 4751's head moved with them. Test surface is 360 added test lines against the 618 the audit measured, and comment plus docstring is 9.9 percent of the 576 added non-blank production lines, measured on `git diff 6c2dadbb3 HEAD -- torchtitan/`. The Results section below is this box's; cells 1 to 7 and 9 of the plan ran here, cell 8 is declined with the reason in `phase13_k3like_48b_posttrain/MOONEP_H100_2026-09-19.md`.
- 2026-09-19, `7c4041c07`: two pieces of surface the 09-19 diff audit named are gone, since this is a draft. The `MoonEPTableBackend` Protocol had one implementation, which did not declare it, and no second user including the tests, so its two annotations now name the backend directly; and `token_padding` only ever carried MoonEP's own `Buffer` default of 128, which no flavor set. `num_sms` stays, because `launch_prefetch` and `launch_grad_reduce` both take it as a required argument and dropping it would put a constant in their call sites. 30 lines out, CPU tests still 16 passed, ufmt clean against the repo's pinned black and usort.
- 2026-09-17 10:00Z: head `84f2704ce` adds the on-device test (2 passed on two H200s). The H200 dp2 / dp4 moonep rows differ from the standard dispatcher at step 1 with a bitwise fresh-cache floor; held out of the body until located (`MOONEP_H200_2026-09-17.md`, 10:00Z).
- 2026-09-17 10:20Z (GPU box): head `3c458bdf1` = the fused transport `904473aef` + lint (ufmt, file end, buffer guard on `grad_reduce_handles`, four unused pyrefly ignores dropped), pushed; CPU tests 17 passed; scoped pre-commit clean but for the box's pyrefly environment errors. cutlass-dsl: the body's Requirements line says 4.6.0 plus MoonEP's `make_fragment` to `make_rmem_tensor` rename in `grad_reduce.py` (one line, `moonep_onbox/h200/moonep_grad_reduce_cutlass46.patch`), which makes every MoonEP suite pass on 4.6.0. Two follow-ups from the audit belong in Limitations: the fp32 `[E, in, out]` grad table per rank (124 GiB per MoE layer at the released shape; the kernel writes one span) and no interleaved pipeline schedules (per-module tables refreshed in forward, re-read in backward). H200 tables on this head are running (`MOONEP_H200_2026-09-17.md`, 10:20Z section).
- 2026-09-17, head is now `904473aef` on `k3_moonep_seam` (pushed; fast-forward on `9e71d1073`): the weight half runs MoonEP's own `launch_prefetch` and `launch_grad_reduce`, each rank holds `[E / R + B]` rows instead of `[E + B]`, and the per-layer cost is two EP-group barriers and no host sync. Full reasoning, the MoonEP facts it rests on and the open items: `phase13_k3like_48b_posttrain/MOONEP_FUSED_TRANSPORT_2026-09-17.md`. Three consequences for filing: the H200 tables measured on `9e71d1073` describe the copy transport and have to be rerun on this head; the Test plan's pass count is dropped until `pytest` runs on a box where `kimi_k3` imports; and the training cells cannot run on cutlass-dsl 4.6.0 at all, because MoonEP's own grad-reduce kernel fails there while Attention Gym's KDA needs at least 4.5, so the body cannot be filed before that is resolved one way or the other. Evidence on the Windows box: the CPU unit passes through an in-process harness and three mutations (offsets drop the slot rows, `reduce_grad` does nothing, the own-row refresh skips a projection) are all caught; no MoonEP kernel has executed yet.
- 2026-09-17: rebased onto `a3a819c67` (two conflicts, `moe.py` and `model.py`, resolved to the branch tip's own state; per-file branch diffs unchanged) and force-pushed; PR 4751 head is `9e71d1073`, mergeable. The H200 run (`phase13_k3like_48b_posttrain/MOONEP_H200_2026-09-17.md`) is on this head; its tables replace the 5060 block below when they land. cutlass-dsl: the body's Requirements line must say 4.6.0 (Attention Gym's KDA needs >= 4.5; MoonEP's own `setup.py` pins 4.4.2 and its grad-reduce DSL tests fail on 4.6.0, which the branch's NVLink table backend does not use).
- Fixed in `cc46bde23` from the 2026-09-14 audit of `610f721bf`: the mesh check is now `dp_shard * cp * tp == ep`, one prefetch slot count (dispatcher config, read by the experts at attach), the expert GEMMs through `GroupedExperts._grouped_mm`, one combine call in the dispatch backward, the pass-through `wire_meshes` override gone, stale docstrings rewritten. Docstring size not re-measured.
- Superseded by the fused transport (`641b18f53`), kept so the older notes below read correctly: on the copy backend the count was three EP-group barriers per MoE layer per step, one in prefetch and two in reduce, against the two that commit `8fef1aa5f`'s message claimed. The branch now has two, one before the prefetch and one before the reduce, since `launch_grad_reduce` fences the ranks inside the kernel, which is what the Design states.
- The 2 x RTX 5060 Ti cells were rerun on `cc46bde23` against main `b21f7d43e` (logs: `Raising_PRs/PR_K3_PARALLELISM/logs_moonep_2026-09-14/`): dp1 standard, dp1 moonep EP=1 and two fresh-cache main runs are bitwise with main; dp2 x ep2 standard is bitwise with main on the shared cache. One of four fresh-cache main dp2 x ep2 runs moved from step 2 (`9.58979` / `13.3125`, then `7.50132` / `9.8750`); a second cold cache and a warm rerun both read the reference, so no mechanism is claimed. The transport itself ran once, 2026-08-28 on 2 x H100 SXM with an NV18 fabric (`phase13_k3like_48b_posttrain/MOONEP_EVIDENCE_2026-08-28.md`, raw logs `raw_h100nvl_0828/`), on the integration tree of that day (`4961dec31`): the same moonep calls (`buffer.dispatch` / `combine`, `create_nvl_single_owner_tensor`, prefetch, `reduce_grad`), since refactored onto core's seams (`8fef1aa5f`, `cc46bde23`). That run had no noise-floor row for the gradient probe and its forced-hot cell is open (the planner saw a balanced histogram, so the hack did not reach it); both are cells of the H100 plan. Fabric rule from the two boxes: NV18 between the pair (NVSwitch fabric) ran, NV6 direct links (09-15) and NVL / SYS (08-28) did not.
- Test counts below are from `cc46bde23`.
- 2026-09-17 diff audit, full findings in `phase13_k3like_48b_posttrain/MOONEP_TEST_PLAN_2026-09-16.md` (section "Diff audit against the #4577 rules"): nothing of MoonEP's own is rebuilt except `prefetch_weight` / `reduce_grad`, which is half the production diff and the source of the three costs the Design now states; the `[E + B]` table is physically dense, which the released config cannot hold (62.0 GiB bf16 plus 124.0 GiB fp32 per MoE layer per rank at R=8, against 13.8 plus 27.6 compacted), the debug flavor hides it, and cell 9 of the plan is the precondition for compacting it; comment plus docstring is 15.3% of the added lines against #4577's 4.3%, so one trim pass before filing; `check_moonep_mesh` should read core's `efsdp` axis instead of recomputing the product. The audit's "35 commits behind main" is closed by the rebase in the note above.
- 2026-09-17, MoonEP source read at `2bd860b` (same audit section): `prefetch_weight` asserts one contiguous `[E + B]` tensor per projection and slices `[:E]` and `[E:]` for the kernel, the README calls that layout a hard requirement and says each of the `E` rows is the home rank's parameter memory mapped through symmetric memory, and the package ships no allocator for the range while its own end-to-end test builds it as a local tensor. So the fused path is a core-level change (expert parameter storage inside a symmetric range, which reaches FSDP, DTensor and the optimizer), not a table-backend swap, and it is blocked twice: no allocator, and MoonEP's own `test_grad_reduce` and `test_e2e` fail on cutlass-dsl 4.6.0, the version Attention Gym's KDA needs (both pass on 4.4.2, which KDA cannot use). This PR keeps the copy backend, which does not touch that kernel and now has the H200 forced-hot evidence; the compaction to the rank's own rows plus its slots, and the two upstream asks on MoonEP, are the follow-ups.
- Test surface before filing (same audit section): the CPU double's dense-reference case is now duplicated by `moonep_onbox/h200/moonep_hot_probe.py` against the real package, so promote that probe into `tests/unit_tests/gpu/` behind a skip when the package or multicast is absent, shrink the double to the dispatcher contract on `DTensorTestBase` with gloo, and keep the capacity test and the four cheap unit tests; about 250 test lines instead of 618.
- 2026-09-15, vast.ai box `103.60.105.169`: 4 x H100 80GB HBM3 (SXM), every pair on 6 NVLinks with no NVSwitch (`nvidia-smi -q` Fabric State N/A). `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` reads 0 on all four and `_SymmetricMemory.has_multicast_support` is False, so MoonEP cannot run there (its buffers assert multicast). The run needs an HGX H100 with NVSwitch (8-GPU board). Building `moonep._C` also needs an nvcc matching torch's CUDA: the cu130 nightly needs a CUDA 13.0 toolkit (the box ships nvcc 12.8, and no cu128/cu129 2.15 nightlies are published).

--- PASTE BEGIN ---

## Summary

Add MoonEP (MoonshotAI/MoonEP, the balanced EP transport of the Kimi K3 report) as a Kimi K3 MoE comm backend, selected with `model_registry(..., moe_comm_backend="moonep")`.

- `MoonEPTokenDispatcher` (`models/common/token_dispatcher.py`, beside DeepEP and HybridEP; the package-facing primitives in `distributed/moonep/moonep.py`): a `BaseEPTokenDispatcher` subclass whose dispatch and combine run MoonEP's kernels on a persistent buffer allocated from `wire_meshes` on the EP group.
- `MoonEPGroupedExperts` (`models/common/moe.py`): a `GroupedExperts` subclass that computes over this rank's `E / R` expert rows followed by its `B` prefetch slots, which MoonEP's prefetch kernel fills, through `GroupedExperts._grouped_mm`.
- `update_ep_token_dispatcher_config` (`models/common/token_dispatcher.py`): fills the static token capacity of every EP dispatcher config that declares `static_token_capacity`, instead of naming DeepEP and HybridEP, and reads `requires_ep` for the EP=1 refusal; `make_token_dispatcher_config` gains `"moonep"`, so backend selection stays in one place; both declare it, so their behaviour is unchanged.
- Tests: CPU checks for what needs neither the package nor a device, and an on-device test that runs the dispatcher and the expert tables with the real package on two GPUs against a dense reference.

## Design

The buffer is sized by the latent width, since the routed experts consume the stream after `routed_down`, and by the static per-rank token count that core now fills after CP and TP have sharded the token axis.

Dispatch and combine are autograd Functions whose backward is the other kernel on the same plan. Routing weights are applied on the torchtitan side, so the router trains through the same path as with the standard dispatcher.

MoonEP's planner picks the experts to copy, recomputed per dispatch. Each rank publishes its own expert rows through `create_nvl_single_owner_tensor` and its slot gradients through `create_nvl_dist_tensor`, and `launch_prefetch` and `launch_grad_reduce` do the two moves, the second fencing the ranks inside the kernel.

A rank computes on its own experts followed by its slots. The grouped GEMM therefore takes the plan's `cu_seqlens` with the rows of experts homed elsewhere dropped, which are empty because a token reaches either its expert's home rank or a rank holding a slot copy.

The package is imported only when an EP mesh exists; with no EP mesh both classes are their parents, so a flavor carrying the config still runs unsharded. The first version keeps expert parameters whole per EP rank, so `efsdp == 1`, and refuses `dp_replicate`.

Requirements and cost:

- Hopper or newer behind an NVSwitch. Every buffer builds a multicast tensor and asserts multicast support ([`buffer.py#L299`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/buffer.py#L299)), the planner writes with `multimem.st` and prefetch uses TMA, so A100, H100 NVL pairs and PCIe cards cannot run it.
- `nvidia-cutlass-dsl` 4.6 with one line changed in MoonEP, `cute.make_fragment` to `cute.make_rmem_tensor` in `grad_reduce.py`: MoonEP pins 4.4.2 while Attention Gym's KDA kernels need 4.6, and with that line every MoonEP suite passes on 4.6, which is how the runs below were taken.
- Both expert dimensions multiples of 128, the prefetch kernel's tile.
- Per rank and projection, `[E / R + B, in, out]` bf16 rows plus one fp32 `[E, in, out]` gradient table of which only this rank's span is written, since `launch_grad_reduce` addresses gradient rows by global expert id.
- Two EP-group barriers per MoE layer and no host sync. The two kernels are called directly rather than through `Buffer.prefetch_weight` and `Buffer.reduce_grad`, which want one contiguous VMM range of `E + B` rows that `moonep.buffer` ships no allocator for.
- Dispatch and combine are not `torch.library` ops like core's DeepEP and HybridEP, so model compile breaks the graph at every dispatch.

## Results

Three things, in order: the training result does not change, the balance contract holds, and on this flavor the transport costs 1.6 percent of step time for a balance it has no use for.

4 x H100 80GB SXM on an NVSwitch fabric, Kimi K3 debug flavor, seed 42, deterministic, MoonEP `2bd860b` with the cutlass line above, torch 2.15 nightly cu126. Every cell starts from one seed checkpoint, each family shares one warm inductor cache, and each floor row is the same cell again on its own fresh cache.

```
torchrun --nproc_per_node=4 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_moonep \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 4 --parallelism.expert_parallel_degree 4
```

Loss and grad norm at steps 1, 3 and 10.

| cell | step 1 | step 3 | step 10 |
| --- | --- | --- | --- |
| `dp2_std` | `12.52567` / `13.5000` | `7.62942` / `9.4375` | `3.33602` / `2.1719` |
| `dp2_moonep_ep1` | `12.52567` / `13.5000` | `7.62942` / `9.4375` | `3.33602` / `2.1719` |
| `dp2ep2_std` | `12.52567` / `13.5000` | `7.61268` / `9.3125` | `3.28545` / `2.1719` |
| `dp2ep2_moonep` | `12.52362` / `13.6250` | `7.76788` / `9.8750` | `3.27265` / `2.0312` |
| `dp2ep2_std_fresh` | `12.52567` / `13.5000` | `7.61268` / `9.3125` | `3.28545` / `2.1719` |
| `dp4_std` | `12.54318` / `13.3750` | `7.79811` / `10.8750` | `3.09682` / `1.8359` |
| `dp4ep4_std` | `12.54318` / `13.3125` | `7.77723` / `11.3125` | `3.17123` / `2.2969` |
| `dp4ep4_moonep` | `12.56253` / `13.2500` | `7.69331` / `11.3750` | `3.17237` / `2.0625` |
| `dp4ep4_std_fresh` | `12.54318` / `13.3125` | `7.77723` / `11.3125` | `3.17123` / `2.2969` |

Both floor rows are bitwise with their reference at every step, so the noise floor here is zero rather than small, and `moe_comm_backend="moonep"` at EP=1 is bitwise with the standard dispatcher, which is what that fallback claims to be.

The step-1 difference in the EP rows is therefore real, and it is located. A per-layer trace puts the origin in the first MoE layer, which takes a bitwise identical input and makes bitwise identical routing decisions and whose routed-expert output still differs by a median 2.6e-3 per token with no token dropped; the next layer's router then flips 6 of its 1024 top-k slots. The operation that differs is the sum of a token's top-k expert copies, which core sums with `deterministic_scatter_add` into a bf16 accumulator and MoonEP in its combine kernel.

Against one fp32 dense reference on the same tokens and weights:

| pair | median rel | max rel |
| --- | ---: | ---: |
| standard against the fp32 reference | `4.649e-03` | `5.648e-03` |
| moonep against the fp32 reference | `4.458e-03` | `5.478e-03` |
| standard against moonep | `2.913e-03` | `3.904e-03` |

The two dispatchers disagree by less than either one's distance to the reference, so neither is the more correct and the difference is which bf16 rounding of the same sum each takes. It becomes visible downstream because a router is a comparison, and a perturbation below the layer's own bf16 floor still flips a near-tie.

So the tables above say MoonEP does not change the result. What it is for is a contract rather than a speedup: every rank receives exactly `S x K` tokens whatever the routing does, guaranteed by reserving `E / R` redundant-expert slots per rank.

The table below checks that this integration preserves it, on the same tokens and weights with only the dispatcher changed, at 256 experts, top-k 8, `D` 1024, four ranks, 4096 tokens per rank, under Zipf routing with the same hot experts on every rank:

| alpha | route max/mean | standard max/mean | moonep max/mean | standard rows | moonep rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1.22 | 1.01 | 1.01 | 131072 | 147584 |
| 1.0 | 25.38 | 1.29 | 1.00 | 131072 | 145920 |
| 2.0 | 32.00 | 1.52 | 1.01 | 131072 | 153984 |

MoonEP holds the per-rank load at 1.00 to 1.02 while the standard path climbs to 1.52. The same sweep carries a fully hot row, every token routed to the experts homed on one rank, where the standard path reads 4.00 with the other three ranks receiving nothing and MoonEP reads 1.00. At 8192 tokens per rank it reads the same.

The cost is the static layout: rows are padded to a multiple of 128 tokens, 12.6 percent more rows at 512 tokens per expert and 5.8 percent at 1024, and 56 percent on the debug flavor whose mean expert receives exactly 128.

Its communication time is flat under imbalance. MoonEP's own benchmark at the report's shapes (`E` 384, `H` 7168, `K` 8, `S` 8192) on four ranks moves the dispatch by 0.4 percent and the combine by 3 percent across a hundredfold change in maxvio.

The static shapes also remove a per-layer host synchronization. `AllToAllTokenDispatcher.dispatch` calls `.tolist()` on both split vectors, two device-to-host transfers per MoE layer, and the MoonEP path has none in the per-layer forward.

End to end on the debug flavor, median forward plus backward per step over steps 2 to 10 on the shared warm cache. These are the numerics cells, which run deterministically without compile at 256 tokens per micro-batch per rank, so they bound the transport's cost in that regime rather than measure its speed:

| cell | median fwd+bwd per step |
| --- | ---: |
| `dp4ep4_std` | 5227.1 ms |
| `dp4ep4_moonep` | 5310.5 ms |
| `dp2ep2_std` | 5162.5 ms |
| `dp2ep2_moonep` | 5194.6 ms |
| `dp2_moonep_ep1` | 4286.3 ms |

MoonEP is 1.6 percent slower at `dp 4 x ep 4` and 0.6 percent at `dp 2 x ep 2`. The EP=1 fallback matches the standard path to 0.3 ms, on the same code path.

That is the expected sign here. This flavor's routing is near uniform, so the balancing has nothing to balance while the barriers and the prefetch copies still cost, and the imbalance it removes at four ranks is only 1.5x. The same sweep run over rank counts, which is arithmetic on the routing with no transport in it, reads 8.3x at 64 ranks with 4 experts each, the regime a K3-shaped expert count runs in.

The comparison the report makes is against DeepEP, whose time is set by the hottest rank, and it is not in this PR yet. So this PR claims correctness, wiring and the balance mechanism, not a speedup.

## Limitations

The gradient table is one fp32 `[E, in, out]` tensor per projection per rank, since `launch_grad_reduce` addresses grad rows by global expert id and writes only this rank's span; at the released expert shape that is a physical row per expert on every rank. An allocation whose other spans are not physical, which is MoonEP's own distributed tensor, or a kernel entry that takes the span would remove it.

Interleaved pipeline schedules are out of scope: the expert tables are per-module state that the forward refreshes and the backward's recompute reads again, so a later micro-batch's forward must not run before an earlier one's backward on the same module.

Expert parallelism must cover a rank's whole expert chunk (`efsdp == 1`) and `dp_replicate` is not wired, both refused with a message.

How much imbalance is left to absorb depends on the configuration, and at this one it is little. On C4 rather than the repeating debug set, with the router bias given a hundred steps to converge, routing reaches a maxvio of 0.31 and the standard dispatcher's rank imbalance is 1.04. The balance the transport guarantees is worth its cost where experts per rank is small, which is the released configuration rather than this one.

## Test plan

```
pytest tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py \
       tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py -q     11 passed
pytest tests/unit_tests/gpu/test_kimi_k3_moon_ep.py -q                   2 passed
```

The CPU file covers what the spec selects, the EP=1 fallback, the import guard, the mesh precondition on real `ParallelDims` meshes with both refusals, and the capacity fill. The import-guard case skips where the package is installed.

The on-device file runs the dispatcher and the expert tables with the real package under forced-hot and uniform routing, checking outputs, input gradients and expert-row gradients against an fp32 reference. It passed on two H200s and on the 4 x H100 box, and skips where the package or multicast is missing.

MoonEP's own suites pass on 2 and 4 ranks on both boxes. The scoped pre-commit hooks pass on the changed files, and `pyrefly` reports the same errors on the branch and on main.

Below the slot bound the model build refuses and names it, `MoonEP needs at least E / R = 8 prefetch slots to place every duplicated expert, got B=7`, rather than hanging. Over one step's 23 MoE layers no layer puts a token in a row homed on another rank, and 19 of them place tokens in a prefetch slot.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
