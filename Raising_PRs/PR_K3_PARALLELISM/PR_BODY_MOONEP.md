### Summary

Adds MoonEP as a Kimi K3 MoE comm backend -- the report's balanced EP transport (sec 5.2.1), selectable as `model_registry("...", moe_comm_backend="moonep")` through the same spec parameter the EP PR introduces. Stacks on that PR; nothing here touches core. The moonep package (MoonshotAI/MoonEP) is an optional dependency like fla and DeepEP, NVLink-mapped GPUs required.

### Design

- One `BaseEPTokenDispatcher` subclass maps titan's dispatch/combine contract onto moonep's `Buffer` (`dispatch -> (hidden, weights, cu_seqlens, plan)`, weights applied combine-side exactly as the standard dispatcher does; the kernels carry their own backward -- dispatch's backward is combine on the same plan and vice versa, expressed as two autograd.Functions). The buffer is sized by the LATENT width: K3's routed experts consume the stream after `routed_down`.
- The expert side computes over moonep's `[E + B]` weight tables: `E` home experts plus `B` prefetch slots that `buffer.prefetch_weight` fills with experts this rank was handed copies of; the grouped GEMM runs over the combined table, and `buffer.reduce_grad` returns every rank's slot gradients to the experts' home ranks in backward. Slot bookkeeping follows `plan.experts_to_copy`.
- Everything stays in the model folder (`moon_ep_dispatcher.py`, `moon_ep_experts.py`); core's dispatcher factory keeps no non-PyTorch dependency. EP=1 falls back to local dispatch without importing moonep.
- The unit is testable WITHOUT the package or NVLink: `moonep_fake.py` implements the Buffer API surface in-process (ranks as threads, collectives as barriers, a test-chosen duplication map), and the CPU tests drive the full path -- dispatch, slots, combine, slot-grad reduction -- against a dense per-token reference, forward and gradients.

### Evidence

2x H100 80GB, NVLink NV18, torch nightly cu130, moonep master @ 2bd860b.

- moonep's own tests at 2 ranks: planning / dispatch (12) / combine (14) / e2e / grad_reduce (12) / prefetch (14) all pass.
- Integration ep2 x fsdp2 trains (12.47486 -> 9.19460 over three steps, deterministic seed).
- Per-parameter gradient probe against the standard dispatcher on the same seed: largest relative differences 2.1-2.6e-1 on 1e-4-magnitude parameters -- ordinary backend arithmetic, no parameter family deviates. (For calibration: a real gradient-loss bug in another backend showed as 0.998 relative difference concentrated on exactly the affected parameters.)
- The copy path is live in real training: with routing forced hot, a slot populates and prefetch/slot-GEMM/reduce-grad all execute; throughput with the machinery active equals the balanced-routing run (345 vs 347 tps).
- Known open item, stated rather than hidden: across two very different forced load shapes the plan's `experts_to_copy` did not change (hot cells: moonep 345 vs standard 336 tps). Reading the moonep planner source (planning.py) resolves most of it: planning is stateless and recomputed per dispatch from that step's all-gathered `tokens_per_expert` (no policy knobs), a rank group is rebalanced only when its load exceeds `S*K`, and each rank copies the top-B remote experts by rebalanced allocation with ties to the lowest index -- so the constant single-copy plan seen under BALANCED routing is the planner's genuine marginal output, while the forced-hot plan we observed is inconsistent with a forced histogram (it should have been rank 1 copying expert 0). That points at the load-forcing hack in those throwaway cells, not at this integration: the call site passes `plan=None` and the per-step histogram derived from the same routing ids. A one-cell retest with the histogram printed at the call site and the expected-plan oracle is queued for the next NVLink session.

CPU tests (no package needed): the fake-world end-to-end with a forced duplication map, spec selection, latent sizing, EP=1 round-trip, import guard.

### Changed files

    torchtitan/models/kimi_k3/
      moon_ep_dispatcher.py    the contract mapping and the two autograd.Functions
      moon_ep_experts.py       [E+B] tables, prefetch, slot-grad reduce
      config_registry.py       the moonep debug flavor
      tests/moonep_fake.py     the in-process Buffer fake
      tests/test_moon_ep_dispatcher.py
