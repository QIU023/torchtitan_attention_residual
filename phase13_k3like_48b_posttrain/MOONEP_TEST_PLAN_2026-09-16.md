# MoonEP: what the code does, what the report claims, and the test plan (2026-09-16)

## The code (`k3_moonep_seam` = `a706a881d`, five commits on main `b21f7d43e`; also on the integration tree)

- `MoonEPTokenDispatcher` (`kimi_k3/moon_ep_dispatcher.py`): a `BaseEPTokenDispatcher` whose dispatch and combine are two autograd Functions over MoonEP's persistent `Buffer` (`buffer.dispatch` / `buffer.combine`, each the other's backward on the same plan). The buffer is built once per EP group from `S` (the static per-rank token count, filled by core's `update_ep_token_dispatcher_config`), `H` (the latent width), `K`, `E`, `num_ep_ranks` and `B`. **The plan of redundant experts is MoonEP's**: `buffer.dispatch` returns it; the torchtitan side never computes a plan.
- `MoonEPGroupedExperts` (`kimi_k3/moon_ep_experts.py`): `GroupedExperts` over `[E + B]` weight tables (the home experts, then `B` prefetch slots); before the routed GEMM `backend.prefetch(plan, tables)` copies the planned experts into the slots, and in backward the `[B]` slot gradients are staged locally and `backend.reduce_grad(plan, grad_tables)` adds them into their home ranks' rows (`MoonEPTableBackendNVLink`, over MoonEP's public primitives). The GEMMs are core's `GroupedExperts._grouped_mm`.
- `B` defaults to `E // num_ep_ranks` (`experts.attach`: `local if slots is None else slots`), the report's bound; a recipe can set `num_prefetch_slots` lower.
- Mesh rule: `dp_shard * cp * tp == ep` (EP spans every rank; `dp_replicate` refused); with no EP mesh both classes are their parents, so `moe_comm_backend="moonep"` at EP=1 runs unsharded.
- Core change: `token_dispatcher.py` fills the static token capacity of any persistent EP backend that declares `static_token_capacity` (DeepEP and HybridEP declare it too, unchanged behaviour).
- CPU tests (12): spec selection and latent sizing, the EP=1 fallback, the import guard, the mesh check, and a two-rank in-process double of the buffer and tables (ranks as threads) against a dense reference with a duplicated expert, forward and gradients, plus core's capacity fill.
- Hardware: every MoonEP buffer builds a multicast tensor and asserts multicast support (`moonep/api.py#L362`, `buffer.py#L299`); that is NVLink SHARP multicast, Hopper or newer behind an NVSwitch. `matrix_scripts/moonep_multicast_probe.py` reads the two flags before any hours are paid for. Not run so far: the transport itself (every box seen lacked NVSwitch: 2 x H100 NVL on 08-28, 4 x H100 SXM direct-NVLink on 09-15, the 5060 Ti box).

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

- Two GPUs (R = 2) reserve 16 slots per rank, half the experts: a skewed router is absorbed without ever pressing the bound, and the reduce path carries one peer. It still proves the transport runs, which no box so far has.
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

What goes in the PR body: cells 2 to 5 as one table (loss steps 1 / 3 / 10, tokens per rank, slot use, step time, allocator retries) and the gradient comparison as a floor statement; cells 6 to 8 as one line each. The body already written (`Raising_PRs/PR_K3_PARALLELISM/PR_BODY_MOONEP.md`) keeps its Summary and Design; its Results section is replaced by the H100 tables when they exist, and it stays a draft while the package is not on the CI boxes.
