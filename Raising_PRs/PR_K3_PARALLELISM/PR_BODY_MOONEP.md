# PR title: [Kimi K3] MoonEP as a MoE comm backend, on the standard dispatcher seam

Branch `k3_moonep_seam` = `7c4041c07`, thirteen commits on main `6c2dadbb3` (2026-09-18), zero behind it; 8 files, +1007/-17, of which 360 lines are tests. Open as draft PR 4751, whose head follows this branch. Earlier heads, for the notes below: `9e71d1073` (rebase onto `a3a819c67`, 2026-09-17), `904473aef` (the fused transport), `3c458bdf1` (lint), `84f2704ce` (the on-device test). Draft until the package is public on the CI boxes; the CPU tests need neither the package nor a GPU. Body in the #4577 format (2026-09-14), Results from the 4 x H100 box (`phase13_k3like_48b_posttrain/MOONEP_H100_2026-09-19.md`); cells in `MOONEP_TEST_PLAN_2026-09-16.md`.

Notes for filing:
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

- `MoonEPTokenDispatcher` (`kimi_k3/moon_ep_dispatcher.py`): a `BaseEPTokenDispatcher` subclass whose dispatch and combine run MoonEP's kernels on a persistent buffer allocated from `wire_meshes` on the EP group.
- `MoonEPGroupedExperts` (`kimi_k3/moon_ep_experts.py`): a `GroupedExperts` subclass that computes over this rank's `E / R` expert rows followed by its `B` prefetch slots, which MoonEP's prefetch kernel fills, through `GroupedExperts._grouped_mm`.
- `update_ep_token_dispatcher_config` (`models/common/token_dispatcher.py`): fills the static token capacity of every EP dispatcher config that declares `static_token_capacity`, instead of naming DeepEP and HybridEP; both declare it, so their behaviour is unchanged.
- Add tests: CPU checks for what needs neither the package nor a device (what the spec selects, the EP=1 fallback, the import guard, the mesh precondition, and the core capacity fill), and an on-device test that runs the dispatcher and the expert tables with the real package on two GPUs against a dense reference.

## Design

The buffer is sized by the latent width, since the routed experts consume the stream after `routed_down`, and by the static per-rank token count that core now fills after CP and TP have sharded the token axis. Dispatch and combine are autograd Functions whose backward is the other kernel on the same plan; routing weights are applied on the torchtitan side, so the router trains through the same path as with the standard dispatcher. MoonEP's planner picks the experts to copy (`plan.experts_to_copy`, recomputed per dispatch); each rank publishes its own expert rows in an NVLink mapping from `create_nvl_single_owner_tensor` and its slot gradients in one from `create_nvl_dist_tensor`, and MoonEP's own `launch_prefetch` and `launch_grad_reduce` do the two moves, the second fencing the ranks inside the kernel. The rows a rank computes on are its own experts followed by its slots, so the grouped GEMM takes the plan's `cu_seqlens` with the rows of experts homed elsewhere dropped, which are empty here because a token reaches either its expert's home rank or a rank holding a slot copy.

The transport stays in the model folder, like fla, and the package is imported only when an EP mesh exists; with no EP mesh both classes are their parents, so a flavor carrying the config still runs unsharded. The first version keeps expert parameters whole per EP rank and refuses `dp_replicate` and any mesh with `dp_shard * cp * tp != ep`.

Requirements and cost: Hopper or newer with NVSwitch. The kernels are CuTeDSL: MoonEP pins `nvidia-cutlass-dsl==4.4.2`, and on 4.6 (the version Attention Gym's KDA kernels need) its grad-reduce kernel calls `cute.make_fragment`, removed in that release; one line of `moonep/grad_reduce.py` (`cute.make_rmem_tensor`) makes every MoonEP suite pass there, which is how the runs below were taken. Every MoonEP buffer builds a multicast tensor ([`moonep/api.py#L362`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/api.py#L362)) and asserts multicast support ([`moonep/buffer.py#L299`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/buffer.py#L299)), the planner writes with `multimem.st` ([`moonep/planning.py#L149-L156`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/planning.py#L149-L156)), and prefetch and the dispatch epilogue use TMA ([`moonep/prefetch.py#L129-L208`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/prefetch.py#L129-L208), [`moonep/dispatch_epilogue.py#L197-L227`](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/dispatch_epilogue.py#L197-L227)), so A100, H100 NVL pairs and PCIe cards cannot run it. Its prefetch kernel tiles 128 by 128, so both expert dimensions have to be multiples of 128. Every rank holds `[E / R + B, in, out]` bf16 rows per projection, and one fp32 `[E, in, out]` gradient table per projection of which it writes only its own span, because `launch_grad_reduce` addresses gradient rows by global expert id. Each MoE layer costs one EP-group barrier before the prefetch and one before the reduce, and no host sync. `Buffer.prefetch_weight` and `Buffer.reduce_grad` take each projection as one contiguous VMM range of `E + B` rows, which `moonep.buffer` ships no allocator for (MoonEP's own end-to-end test builds that range as a local tensor), so the two kernels are called directly, as MoonEP's `test_grad_reduce` does, and the reduce's barrier handles come from the Buffer's context. Dispatch and combine are not `torch.library` ops like core's DeepEP / HybridEP, so model compile breaks the graph at every dispatch.

## Results

4 x H100 80GB SXM on an NVSwitch fabric (NV18 between every pair, multicast on all four), Kimi K3 debug flavor, seed 42, deterministic, MoonEP `2bd860b` with the cutlass 4.6 line above, torch 2.15 nightly cu126.

```
torchrun --nproc_per_node=4 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_moonep \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 4 --parallelism.expert_parallel_degree 4
```

Loss and grad norm at steps 1, 3 and 10. Every cell starts from one seed checkpoint, each family shares one warm inductor cache, and each floor row is the same cell again on its own fresh cache.

    cell                  step 1             step 3             step 10
    dp2_std               12.52567/13.5000   7.62942/9.4375     3.33602/2.1719
    dp2_moonep_ep1        12.52567/13.5000   7.62942/9.4375     3.33602/2.1719
    dp2ep2_std            12.52567/13.5000   7.61268/9.3125     3.28545/2.1719
    dp2ep2_moonep         12.52362/13.6250   7.76788/9.8750     3.27265/2.0312
    dp2ep2_std_fresh      12.52567/13.5000   7.61268/9.3125     3.28545/2.1719
    dp4_std               12.54318/13.3750   7.79811/10.8750    3.09682/1.8359
    dp4ep4_std            12.54318/13.3125   7.77723/11.3125    3.17123/2.2969
    dp4ep4_moonep         12.56253/13.2500   7.69331/11.3750    3.17237/2.0625
    dp4ep4_std_fresh      12.54318/13.3125   7.77723/11.3125    3.17123/2.2969

Both floor rows are bitwise with their reference at every step, so the noise floor for these cells is zero rather than small, and `moe_comm_backend="moonep"` at EP=1 is bitwise with the standard dispatcher, which is what that fallback claims to be.

With a zero floor the MoonEP rows differ from the standard ones at step 1, so that difference is real and it is located rather than attributed. A step-1 per-parameter gradient comparison at `dp 4 x ep 4` (726 parameters, the same seed checkpoint, one shared warm cache) reproduces both cells exactly, `12.543177` and `12.562533`, and its floor row, the standard cell against itself on a fresh cache, is bitwise on all 726 parameters, the loss and the total gradient norm. The routed expert weights are the least affected of the eight parameter groups (2.65e-2 maximum relative difference against 3.14e-1 for attention, whose largest entries carry gradient norms of 1e-4).

A per-layer trace of both runs puts the origin in one place. The first MoE layer receives a bitwise identical input and makes bitwise identical routing decisions (zero top-k flips, scores identical), and its routed-expert output differs, with no token dropped and a per-token relative difference of median 2.6e-3. The next layer's router then flips 6 of its 1024 top-k slots, and from there the runs route different tokens to different experts.

The operation that differs is the sum of a token's top-k expert copies. Both paths apply the routing weights identically (`routed_output.to(float32) * weights`, rounded back to bf16); core then sums the copies with `deterministic_scatter_add` into a bf16 accumulator, while MoonEP hands them to its own combine kernel. Two controls size that. The standard path is bitwise invariant to a layout change that reorders the same arithmetic: at `ep 1` instead of `ep 4`, with a different dispatcher, different grouping and different grouped-GEMM offsets, the first MoE layer's output is identical on all 131072 elements. And both dispatchers, run on the same tokens and expert weights against one fp32 dense reference at the on-device test's shapes (2 ranks, 256 tokens, $D = 512$, 32 experts, top-k 4), sit the same distance from it:

    pair                          median rel   max rel
    standard vs fp32 reference     4.649e-03   5.648e-03
    moonep vs fp32 reference       4.458e-03   5.478e-03
    standard vs moonep             2.913e-03   3.904e-03

The disagreement between the two dispatchers is smaller than either one's distance to the reference, so neither output is the more correct one and the difference is which of two bf16 roundings of the same sum each path takes. It becomes visible downstream because a router is a comparison: a perturbation below the layer's own bf16 floor still flips a near-tie, and the flip is discrete.

The tables above establish that MoonEP does not change the result. What it is for is stated in the Kimi K3 report as a contract rather than a result: every rank receives exactly `S x K` tokens whatever the routing does, guaranteed by reserving `E / R` redundant-expert slots per rank. So the table below is not a measurement of benefit, it is the check that this integration preserves that contract, taken on the same tokens and expert weights with only the dispatcher changed, at 256 experts, top-k 8, $D = 1024$, four ranks, under Zipf routing with the same hot experts on every rank:

    4096 tokens per rank, 512 per expert
     alpha  route max/mean  standard max/mean  moonep max/mean  standard rows  moonep rows
       0.0           1.22               1.01             1.01         131072       147584
       1.0          25.38               1.29             1.00         131072       145920
       2.0          32.00               1.52             1.01         131072       153984

    8192 tokens per rank, 1024 per expert
       0.0           1.17               1.01             1.00         262144       277248
       2.0          32.00               1.53             1.00         262144       283136

MoonEP holds the per-rank load at 1.00 to 1.02 across the sweep while the standard path climbs to 1.52, and at the extreme, every token routed to the experts homed on one rank, the standard path reads 4.00 with three of four ranks idle and MoonEP reads 1.00 with the same total row count.

The cost is the static layout: MoonEP pads every row's token count to a multiple of 128, which is 12.6 percent more rows at 512 tokens per expert and 5.8 percent at 1024. On the debug flavor, whose mean expert receives exactly 128 tokens, the same figure is 56 percent, so it is a function of how many padding units a row carries rather than a property of the transport.

Two more of the transport's stated properties are checkable here. Its communication time is flat under imbalance: its own benchmark, run at the report's shapes (`E = 384`, `H = 7168`, `K = 8`, `S = 8192`) on four ranks rather than the eight it asserts, reads

    maxvio    planning   dispatch fwd   combine fwd   prefetch
      0.20    104.3 us      2041.0 us     1426.3 us    499.7 us
      1.01    104.0 us      2042.2 us     1411.9 us    501.7 us
     10.04    105.4 us      2034.0 us     1420.1 us    500.0 us
     19.79    101.9 us      2039.0 us     1449.0 us    501.1 us

so across a hundredfold change in maxvio the dispatch moves 0.4 percent and the combine 3 percent, with planning at 5 percent of the dispatch and the weight prefetch at 24.6, both constant. And the shapes being static removes a per-layer host synchronization: `AllToAllTokenDispatcher.dispatch` calls `.tolist()` on both split vectors, two device-to-host transfers per MoE layer, and the MoonEP path has none in the per-layer forward.

What none of this shows is that the insensitivity is worth anything, because that claim is comparative. The report and MoonEP's own benchmark both make it against DeepEP, whose time is set by the hottest rank, and that comparison is not in this PR: DeepEP v2's `ElasticBuffer` asserts NCCL GIN at construction, GIN needs an RDMA device, and the box these numbers come from has none. Against the dispatcher this repo ships, the rank-level imbalance MoonEP removes at four ranks is 1.5x, and that figure grows as experts per rank falls, which is arithmetic on the routing with no transport involved:

     ranks   experts/rank   alpha=1.0   alpha=2.0
         4             64        1.29        1.53
        16             16        2.30        2.39
        64              4        6.93        8.28


End to end on the debug flavor, median forward plus backward per step over steps 2 to 10 on the shared warm cache:

    cell              median fwd+bwd per step
    dp4ep4_std                     5227.1 ms
    dp4ep4_moonep                  5310.5 ms
    dp2ep2_std                     5162.5 ms
    dp2ep2_moonep                  5194.6 ms
    dp2_std                        4286.0 ms
    dp2_moonep_ep1                 4286.3 ms

MoonEP is 1.6 percent slower at `dp 4 x ep 4` and 0.6 percent at `dp 2 x ep 2`, and the EP=1 fallback matches the standard path to 0.3 ms, which is the same code path. That is the expected sign here: this flavor's routing is near uniform, so the balancing has nothing to balance while the two EP-group barriers and the prefetch copies still cost. A size where routing is genuinely imbalanced is not something one node can show, so this PR claims correctness, wiring and the balance mechanism, not a speedup.

## Limitations

The gradient table is one fp32 `[E, in, out]` tensor per projection per rank, since `launch_grad_reduce` addresses grad rows by global expert id and writes only this rank's span; at the released expert shape that is a physical row per expert on every rank. An allocation whose other spans are not physical (MoonEP's own distributed tensor) or a kernel entry that takes the span would remove it.

Interleaved pipeline schedules are out of scope: the expert tables are per-module state that the forward refreshes and the backward's recompute reads again, so a later micro-batch's forward must not run before an earlier one's backward on the same module.

Expert parallelism must cover the whole expert chunk of a rank (`efsdp == 1`) and `dp_replicate` is not wired, both refused with a message.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py -q` (11 passed; the import-guard case skips where the package is installed): spec selection and latent sizing, the EP=1 local fallback, the import guard, and the mesh precondition against real `ParallelDims` meshes on four ranks, covering the accepted case and both refusals; the capacity fill after CP and TP, the EP=1 refusal, the divisibility check, the local fallback, a backend without a static capacity. The comparison against a dense reference lives in the on-device test below, where it runs with the real package instead of a double.
- The same two files with `tests/unit_tests/cpu/test_inference_moe.py` added, `pytest tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py tests/unit_tests/cpu/test_inference_moe.py -q` (16 passed, the 11 above plus that file's 5); the scoped pre-commit hooks (flake8, ufmt, end-of-file, codespell, pydoclint) pass on the changed files, and `pyrefly` reports the same errors on the branch and on main, none in these files.
- 4 x H100 80GB SXM (NV18, NVSwitch, multicast on every GPU), moonep `2bd860b` with the cutlass 4.6 line above, torch 2.15 nightly cu126: moonep's own suites pass on 2 and on 4 ranks (planning, dispatch, combine, grad_reduce, prefetch, e2e), and the cells of the Results section run from one seed checkpoint on this box.
- Forced-hot routing (every token to one rank's experts) at `dp 4 x ep 4`: the run trains and the other ranks' prefetch slots fill. With the slot count set below the bound, planning refuses it at model build time and names the bound, `MoonEP needs at least E / R = 8 prefetch slots to place every duplicated expert, got B=7` and the same for `B=1`, rather than hanging.
- 4 x H200 (NV18, NVSwitch, multicast on every GPU), moonep `2bd860b` with the cutlass-4.6 line above, torch 2.15 nightly cu126: moonep's own suites on two ranks all pass (planning 18 passed 1 skipped, dispatch 12, combine 14, e2e 1, grad_reduce 12, prefetch 14).
- `pytest tests/unit_tests/gpu/test_kimi_k3_moon_ep.py -q` (2 passed on two H200s, 2 passed on the 4 x H100 box): the dispatcher and the expert tables with the real package, forced-hot routing (every token to the experts homed on one rank, so the other rank's prefetch slots fill) and uniform routing, outputs, input gradients and expert-row gradients against an fp32 reference on the gathered tokens; skipped where the package or multicast is missing.
- Table occupancy at `dp 4 x ep 4` on the debug model, all 23 MoE layers of one step: no layer puts a token in a row belonging to an expert homed on another rank (0 of 23), this rank's own expert rows carry 1024 to 1664 tokens per layer, and 19 of the 23 layers place tokens in a prefetch slot. So 16 of the 40 physical rows, `E / R` experts plus `B` slots, are the whole working set.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
