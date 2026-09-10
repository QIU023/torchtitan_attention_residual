# PR title (draft): [Kimi K3] A DistMuon recipe with per-head MLA and KDA compute layouts

Draft PR from fork branch `k3_dist_muon` (one commit on main `ac10ca48f`; independent of the parallelism PRs). It waits for #4353 (DistMuon under tensor parallelism): the recipe refuses `tensor_parallel_degree > 1` until then, as the Kimi K2.5 recipe does. Every number below is from 8 x RTX 5060 Ti (SM120; the KDA capability guard lifted locally for the runs, not in the commit).

--- PASTE BEGIN ---

## Summary

`kimi_k3_debugmodel_muon` trains the Kimi K3 debug model with DistMuon on its matrix parameters and AdamW on the rest, after the Kimi K2.5 recipe. Muon orthogonalises per head where a weight stacks heads along its output dimension, per expert on the routed experts, and whole everywhere else.

- Per-head `BlockShard(dim=0, block_size=head)`: the MLA query up-projection `wq_b` (block `qk_nope_head_dim + qk_rope_head_dim`), the MLA key-value up-projection `wkv_b` (block `qk_nope_head_dim + v_head_dim`), and the KDA `q_proj` / `k_proj` / `v_proj` (block `head_dim`).
- Per expert: the routed experts' `w1_EFD` / `w2_EDF` / `w3_EFD`, through the Kimi K2.5 registry's `_per_expert_compute_layout`, rebuilt from the final parallelism config in `__post_init__` because the command line can still override `expert_parallel_degree` (which decides between the 1-D `dp_shard` layout and the 2-D EP/EFSDP one).
- Owned whole: the MLA compressions `wq_a` / `wkv_a`, `wo` and the attention gate; the KDA `output_gate`, `output_proj`, `beta`, `forget_a`, `forget_b`; the latent projections `routed_up` / `routed_down`; the shared experts; the router gate; the dense feed-forward.
- AdamW: norms, biases, the KDA convolutions and gate parameters (`A_log`, `dt_bias`), the one-row attention-residual projections, embeddings, the LM head and the vision tower.
- Buckets: one per pair of layers, with the pair's routed experts in a bucket of their own, as the K2.5 recipe does.

## Implementation

All in `torchtitan/models/kimi_k3/config_registry.py`: `_dist_muon_optimizer` builds the FQN-keyed `compute_sharding_by_fqn` and the bucket list from the model config (so a different layer mix or head shape changes the layouts with it), `_align_dist_muon_expert_compute_layouts` rebuilds the per-expert layouts from the final `ParallelismConfig`, and `_KimiK3MuonTrainerConfig.__post_init__` applies that and refuses tensor parallelism. The Muon pattern matches the matrices by FQN suffix; the AdamW group takes the rest. lr 8e-4 on both, `match_rms_adamw`, weight decay 0.1, the K2.5 debug settings.

## Limitations

- Tensor parallelism is refused until DistMuon supports its storage layouts (#4353).
- Pipeline and context parallelism are not exercised (the recipe is a debug recipe; the layouts are per-FQN and do not depend on the split).
- No learning-rate study: the recipe reuses the K2.5 debug recipe's lr, and the descent at that lr is the optimizer's.

## Tests

```text
pytest tests/unit_tests/cpu/test_integration_test_definitions.py
```

Result: 13 passed. pre-commit on the touched file: every hook passes except the project-wide `pyrefly-check`, which reports the same 58 pre-existing environment errors (missing `transformers` / `torch_checkpointing` imports) on main `ac10ca48f` itself; the error set is identical with and without this commit (none in the touched file).

Layout coverage on the debug model (24 layers, built on the meta device, the Muon pattern applied to `named_parameters`): 750 parameters, 390 match the Muon pattern and 390 have a compute layout, none missing in either direction (66 per-head block shards, 69 per-expert, 255 owned), 360 tensors on AdamW, 24 buckets. DistMuon itself raises on a Muon parameter without a layout, so every run below also checks the first direction.

## Results

`kimi_k3_debugmodel_muon` against the AdamW debug recipe `kimi_k3_debugmodel` on the same cell: `seed=42`, deterministic, 4096 tokens per step, one seed checkpoint per cell (model weights seeded, optimizer state fresh), 3 steps, spmd_types backend, 8 x RTX 5060 Ti. Cells with different data-parallel degrees read different data, so rows compare only within a cell.

| cell | optimizer | step 1 loss | step 2 loss | step 3 loss | step 1 grad norm | step 2 | step 3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | DistMuon | `12.53584` | `11.77823` | `10.55618` | `14.6250` | `15.6875` | `20.5000` |
| dp1 | AdamW | bitwise | `9.49238` | `6.60885` | bitwise | `16.2500` | `9.7500` |
| fsdp2 | DistMuon | `12.52818` | `11.74061` | `10.62273` | `15.0625` | `16.6250` | `17.1250` |
| fsdp2 | AdamW | bitwise | `9.61908` | `7.18662` | bitwise | `14.1875` | `11.0000` |
| ep2 x fsdp2 | DistMuon | `12.52733` | `11.74126` | `10.57633` | `15.0625` | `16.5000` | `19.1250` |
| ep2 x fsdp2 | AdamW | bitwise | `9.62503` | `7.15118` | bitwise | `14.2500` | `12.1875` |

Step 1 is bitwise the AdamW value in every cell (the loss precedes the first update, so the model and data path are untouched by the recipe); DistMuon initialises on the DTensor layouts of all three meshes and trains through three steps. The slower descent at this learning rate is the optimizer's, as the Kimi K2.5 debug recipe shows at the same setting; a learning-rate study is not this recipe's business.

--- PASTE END ---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
