MinimalAsyncEP's `dispatch` returns one of its two recycled receive buffers as the routed expert input. The routed experts' grouped GEMM saves that tensor for its weight gradient, and in a layer's backward the combine backward dispatches the output gradient into the same slot first (dispatch takes slot k, combine k+1, combine-backward k+2 == k with two slots), so `dW1` and `dW3` are computed from gradient rows instead of the expert input. `deepseek_v3_debugmodel_minimal_async_ep` gets exactly zero `w13` gradients under ep2, and the plain `GroupedExperts` exactly zero `w1_EFD` / `w3_EFD`; the total gradient norm hides it (4.44490 against 4.44487 with the standard dispatcher on the debug model) and the integration test only checks that training runs.

This clones the dispatched rows in `MinimalAsyncEPTokenDispatcher.dispatch` when grad is enabled. Inference keeps the view; the combine output is consumed by the combine backward before any later slot write, so it is left alone. The compiled path shows the same thing: `graph_trainer_deepseek_v3_debugmodel_minimal_async_ep` (aot_fx_trace, `--compile.memory_policy full`) has `w13` gradients of exactly zero before and 0.01149 / 0.00811 / 0.00638 after, the values the standard dispatcher gives.

Per-parameter gradient norms, one microbatch from the same seed init, `deepseek_v3_debugmodel` with the plain `GroupedExperts`, ep2 x fsdp2, full activation checkpointing, 2 x H100:

| | routed experts w1/w3 (max rel. diff vs standard) | every other group |
|---|---|---|
| before | 1.00 (all exactly 0) | < 2.5e-3 |
| after | 3.1e-4 | < 2.5e-3 |

Same configuration, 10 steps, seed 42, `--debug.deterministic` (the loss barely moves either way on the debug model, which is why the gradient table is the evidence: at step 2 minimal_async_ep is 2.6e-3 from standard before and 9e-5 after):

| dispatcher | patch | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| standard | - | 8.00752 | 5.07747 | 3.95123 |
| minimal_async_ep | before | 8.00751 | 5.07892 | 3.95201 |
| minimal_async_ep | after | 8.00751 | 5.07750 | 3.95003 |
