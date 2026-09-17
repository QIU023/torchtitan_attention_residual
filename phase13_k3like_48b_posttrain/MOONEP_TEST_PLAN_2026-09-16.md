# MoonEP: what the code does, what the report claims, and the test plan (2026-09-16)

## The code (`k3_moonep_seam` = `a706a881d`, five commits on main `b21f7d43e`; also on the integration tree)

- `MoonEPTokenDispatcher` (`kimi_k3/moon_ep_dispatcher.py`): a `BaseEPTokenDispatcher` whose dispatch and combine are two autograd Functions over MoonEP's persistent `Buffer` (`buffer.dispatch` / `buffer.combine`, each the other's backward on the same plan). The buffer is built once per EP group from `S` (the static per-rank token count, filled by core's `update_ep_token_dispatcher_config`), `H` (the latent width), `K`, `E`, `num_ep_ranks` and `B`. **The plan of redundant experts is MoonEP's**: `buffer.dispatch` returns it; the torchtitan side never computes a plan.
- `MoonEPGroupedExperts` (`kimi_k3/moon_ep_experts.py`): `GroupedExperts` over `[E + B]` weight tables (the home experts, then `B` prefetch slots); before the routed GEMM `backend.prefetch(plan, tables)` copies the planned experts into the slots, and in backward the `[B]` slot gradients are staged locally and `backend.reduce_grad(plan, grad_tables)` adds them into their home ranks' rows (`MoonEPTableBackendNVLink`, over MoonEP's public primitives). The GEMMs are core's `GroupedExperts._grouped_mm`.
- `B` defaults to `E // num_ep_ranks` (`experts.attach`: `local if slots is None else slots`), the report's bound; a recipe can set `num_prefetch_slots` lower.
- Mesh rule: `dp_shard * cp * tp == ep` (EP spans every rank; `dp_replicate` refused); with no EP mesh both classes are their parents, so `moe_comm_backend="moonep"` at EP=1 runs unsharded.
- Core change: `token_dispatcher.py` fills the static token capacity of any persistent EP backend that declares `static_token_capacity` (DeepEP and HybridEP declare it too, unchanged behaviour).
- CPU tests (12): spec selection and latent sizing, the EP=1 fallback, the import guard, the mesh check, and a two-rank in-process double of the buffer and tables (ranks as threads) against a dense reference with a duplicated expert, forward and gradients, plus core's capacity fill.
- Hardware: every MoonEP buffer builds a multicast tensor and asserts multicast support (`moonep/api.py#L362`, `buffer.py#L299`); that is NVLink SHARP multicast, Hopper or newer behind an NVSwitch. `matrix_scripts/moonep_multicast_probe.py` reads the two flags before any hours are paid for. The transport ran once: 2 x H100 SXM with an NV18 fabric on 08-28 (`MOONEP_EVIDENCE_2026-08-28.md`, `raw_h100nvl_0828/`: moonep self-tests 18 / 12 / 14 / 12 / 14 passed plus e2e, the ep2 x fsdp2 smoke `12.47486` then `9.19460`, the gradient probe against standard with no routed-expert parameter among the differences; no floor row, forced-hot cell open). The boxes without multicast were the 2 x H100 NVL pair of 08-28 (`topo -m` SYS), the 4 x H100 SXM of 09-15 (NV6 direct links) and the 5060 Ti box. Renting criterion: `nvidia-smi topo -m` shows NV18 between every pair (an NVSwitch fabric), not NV6, and `moonep_multicast_probe.py` reads True.

## The report (section 5.2.1) against the code

| report claim | where it lives | how the plan below tests it |
| --- | --- | --- |
| "every rank receives exactly S x K tokens, so all ranks perform identical amounts of computation" | MoonEP's planner (`buffer.dispatch`), our static `S` | log the per-rank routed token count after the plan on every layer and step; the bar is equality across ranks on every layer, including under a forced-hot router |
| "at most E/R redundant experts per rank suffice; the bound is essentially tight" | `B = E // R` default | at `B = E/R` the forced-hot runs never fail to plan; at `B = E/R - 1` a run with a hot enough router must either still plan (bound not tight at this scale) or raise, and the log says which; no silent cap |
| "plan from the router outputs of the current micro-batch and layer, prefetch before the routed-expert computation" | `_MoonEPExpertFunction.forward` calls `backend.prefetch(plan, tables)` before the GEMM | the duplicated-expert double already checks the forward; on GPU the ep4 run's slot occupancy is logged per layer (how many slots the plan filled) |
| "stage their gradients in a local reduce buffer and, once the computation completes, reduce them back to the gradient buffers of their home ranks" | `[B]` slot grads plus `backend.reduce_grad` | per-parameter expert-weight gradients of the ep4 MoonEP run against the standard dispatcher on the same seed and batch (same data per rank, same S), reported per group with the floor: the two transports sum the same terms in another order, so bitwise is not expected; the bar is every group inside the bf16 floor of the standard dispatcher against itself under another reduction order (the same cell twice on fresh caches) |
| conventional EP: "dynamically varying shapes of routed-expert activations cause substantial memory fragmentation" | static `S` per rank, persistent buffer | allocator statistics per step (`torch.cuda.memory_stats`: reserved minus allocated, `num_alloc_retries`, segment count) for MoonEP against the standard dispatcher under the forced-hot router; the claim is fewer retries and a flat reserved curve, not a smaller peak at this model size |
| throughput | | step time under the forced-hot router, MoonEP against standard, both after a warm-up step; a debug-model number, not the report's |

## GPU count

Four H100 SXM on one HGX board (NVSwitch), EP = 4 over the whole node (`dp_shard = 4`, `ep = 4` by the mesh rule), the debug model's 32 routed experts, so `E/R = 8` slots per rank.

- Two GPUs (R = 2) reserve 16 slots per rank, half the experts: a skewed router is absorbed without ever pressing the bound, and the reduce path carries one peer. The 08-28 pair already showed the transport running; two ranks add nothing to that, four ranks put the planner in a regime with several slots per rank.
- Eight GPUs (R = 8, 4 slots) is the regime of the real config and double the rental for the same planner code; nothing in the cells below needs more than four ranks, and the imbalance a forced-hot router creates at R = 4 already needs several slots per rank.
- Four is also enough for MoonEP's own two-rank tests and for a 2-rank cell inside the 4-GPU rental (`dp_shard = 2, ep = 2` on two of the four).

Renting a 4-GPU slice of an HGX node is fine only if the probe reads multicast True on the slice; if a provider cannot say, take the 8-GPU node and use four.

## Cells (all with `--debug.seed 42 --debug.deterministic`, one warm step then the measured steps on one inductor cache, the seed checkpoint of `mx4.sh`)

1. `moonep`'s own test suite on 2 and 4 ranks (the package's correctness on the box).
2. ep4 x fsdp4, standard dispatcher, 10 steps: the reference row (and a second run on a fresh cache for the floor).
3. ep4 x fsdp4, MoonEP with `B = E/R = 8`, 10 steps: loss trajectory, the per-rank token counts per layer (equal, the report's claim), slot occupancy per layer, step time, allocator statistics.
4. Cell 3's per-parameter gradient comparison against cell 2 at step 1 (every expert weight, every non-expert weight; non-expert weights are expected bitwise since only the expert transport changed).
5. Forced-hot router (a bias on the router logits that sends most tokens to one or two experts) for cells 2 and 3: the balance claim under real imbalance, the slot count it takes, and the memory statistics where the standard dispatcher's shapes swing most.
6. `B = E/R - 1` and `B = 1` under the forced-hot router: does planning fail, and how does it fail (an error naming the bound, not a hang).
7. ep2 x fsdp2 on two of the four GPUs with MoonEP, 3 steps: the smallest cell, for the PR's CI story.
8. 33-layer debug flavor, ep4, MoonEP, 3 steps: more layers per step, no new claim.
9. From the audit below, before cell 3 is trusted: `matrix_scripts/moonep_row_occupancy_probe.py` (run in place of `torchtitan.train` with the cell-3 flags and `--training.steps 1`) dumps `cu_seqlens` and `plan.experts_to_copy` for the first four dispatches and checks that every row of an expert whose home is another rank carries no tokens. That is the condition under which the `[E + B]` table can be compacted to the rank's own `E / R` rows plus its `B` slots; the in-process double has the property by construction, which is not evidence.

What goes in the PR body: cells 2 to 5 as one table (loss steps 1 / 3 / 10, tokens per rank, slot use, step time, allocator retries) and the gradient comparison as a floor statement; cells 6 to 8 as one line each. The body already written (`Raising_PRs/PR_K3_PARALLELISM/PR_BODY_MOONEP.md`) keeps its Summary and Design; its Results section is replaced by the H100 tables when they exist, and it stays a draft while the package is not on the CI boxes.

## Diff audit against the #4577 rules (2026-09-17, on the question of why the diff is this large)

`b21f7d43e..a706a881d` is +1256 / -17 over 8 files, and it splits almost evenly: 638 lines of production code (dispatcher 255, experts 321, core seam 16, model-folder wiring 46) against 618 lines of tests and the in-process double.

| file | total | code | comment | docstring |
| --- | ---: | ---: | ---: | ---: |
| `kimi_k3/moon_ep_dispatcher.py` | 255 | 172 | 16 | 31 |
| `kimi_k3/moon_ep_experts.py` | 321 | 231 | 15 | 33 |
| `tests/unit_tests/cpu/kimi_k3_moonep_fake.py` | 306 | 229 | 9 | 15 |
| `tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py` | 225 | 163 | 14 | 14 |
| `tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py` | 87 | 50 | 5 | 1 |

Yardstick: core spends more per EP backend than this PR does. DeepEP is a 107-line dispatcher class in `models/common/token_dispatcher.py` plus 605 lines in `torchtitan/distributed/deepep/deepep.py`; HybridEP is 130 lines plus 550 in `hybridep.py`. This PR's transport is 255 plus 321 and adds no file outside the model folder, so the production half is not out of proportion for a fourth backend.

Nothing of MoonEP's own is rebuilt, checked call by call: the planner is MoonEP's (`buffer.dispatch` returns the plan; the torchtitan side only carries it and hands it back), so are the permute, unpermute and transport (`buffer.dispatch` / `buffer.combine` and their reverse pairing in the two autograd Functions); the expert GEMMs run core's `GroupedExperts._grouped_mm`; the static capacity comes from core's `update_ep_token_dispatcher_config`, extended by a declaration on the config instead of a name list, which also deleted 18 lines the model config used to carry; the EP group is `ep_mesh.get_group()`; EP=1 calls core's `LocalTokenDispatcher.dispatch` and `.combine` directly.

One half of MoonEP is rebuilt, and it is where the size sits: `prefetch_weight` and `reduce_grad`. Their own kernels work (08-28 on the NV18 pair: `test_prefetch` 14 passed, `test_grad_reduce` 12 passed), and the reason recorded on 08-28 in `MOONEP_DRAFT.md` still holds: the fused kernels want each projection as one contiguous VMM range whose first `E` rows are the peers' expert chunks mapped in place, and `moonep.buffer` offers `create_nvl_dist_tensor` (equal chunks, no slot tail) and `create_nvl_single_owner_tensor` (a single owner per tensor), neither of which hands out that range. `MoonEPTableBackendNVLink` (about 120 lines) and the recompute inside `_MoonEPExpertFunction` stand in for the two kernels.

Three costs follow, and each one contradicts a sentence of report 5.2.1 that the PR quotes as its motivation:

- The compute tables are local `torch.zeros(E + B, in, out)` (`alloc_weight_table`, and fp32 in `alloc_grad_table`), so every rank physically holds a row for every expert instead of its own `E / R` plus `B`. Released config (latent 3584, expert hidden 3072, 896 experts), per MoE layer per rank: at R=8, 62.0 GiB bf16 plus 124.0 GiB fp32 against 13.8 plus 27.6 compacted (4x); at R=64, 56.0 plus 112.0 against 1.72 plus 3.45 (32x). The debug flavor hides it completely (54 MiB plus 108 MiB per layer), so the cells above run either way, but "at most E/R redundant experts per rank" is a memory claim and this layout does not keep it.
- Three EP-group barriers and two `plan.experts_to_copy.tolist()` host syncs per MoE layer per step, against "sync-free execution ... eliminates the per-layer MoE host synchronization".
- Copies into and out of the slot rows, against "zero-copy", plus one extra expert forward per step, because the forward runs under `no_grad` and the backward recomputes `_compute` with the tables as leaves.

Two ways out, to decide on the box:

(a) Compact the table to `E / R + B` rows and drop the empty groups from `cu_seqlens` (cell 9 above is the precondition). Tokens for an expert reach either its home rank or a rank holding a slot copy, so the rows of other ranks' home experts are empty on this rank and `torch._grouped_mm` returns the same result over the compacted pair. This removes the memory finding only, and it keeps the barriers.

(b) Build the contiguous range (the draft's option 2: reserve a VMM range and map the R chunks and the slot pages back to back) and call `prefetch_weight` / `reduce_grad`. This removes all three costs, deletes most of the table backend, and is the version the report describes.

Smaller findings:

- Comment plus docstring is 15.3% of the added non-blank lines (59 comment, 94 docstring, 845 code) against #4577's 4.3% (8 plus 10 on 398). The dispatcher is densest at 47 on 172. One trim pass before filing.
- `check_moonep_mesh` recomputes `dp_shard * cp * tp == ep`, which core already names: `efsdp` is a mesh axis of size `dp_shard * cp * tp // ep` (`distributed/parallel_dims.py:206` and `:284`). Read core's axis instead of the product.
- `KimiLatentMoE.parallelize` imports the two MoonEP modules inside the function, with a comment saying `moon_ep_experts` imports this module. It does not: it imports `models/common/moe.py` and `moon_ep_dispatcher`. If the cycle is the package `__init__`, the comment should say that; check with a plain module-level import on a box where `kimi_k3` imports (not this one, CuTeDSL).
- The double re-implements MoonEP's dispatch layout (per `(src, token, k)` placement, row padding, `cu_seqlens`) plus a thread-barrier collective, 306 lines. Core unit tests neither DeepEP nor HybridEP; it covers them with `tests/integration_tests/h100.py` cells (`deepseek_v3_fsdp+hybridep+compile`, `qwen3_fsdp+deepep`). The double stays, since it is the only thing that runs without an NVSwitch box and it caught the contract errors of 08-28, but the PR's test plan should also name the h100 cell this backend would add once the package is on the CI boxes.
- The branch is 35 commits behind current main and conflicts in one file (`kimi_k3/moe.py`), from the quantile-balancing merge that also touches `KimiLatentMoE`.
