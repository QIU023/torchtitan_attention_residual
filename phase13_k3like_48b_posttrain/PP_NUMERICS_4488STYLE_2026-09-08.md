# PP numerics in the maintainers' format (2026-09-08)

The DSV3 MTP pipeline PR on upstream (PR 4488) reports its numerics as one table: 1 GPU vs 2-GPU PP, loss and total gradient norm, relative difference per step, max over 5 steps. This note produces the same table for the AttnRes pipeline on the client branch, plus the two control rows a reader needs to interpret the bf16 rows.

## Tree and protocol

- Tree: fork branch `pp_runtime_client` = `464421e13` (the K3 AttnRes stage as a client of the pipeline runtime of PR 4486; 18 commits over that PR's head `713a6bbb6`). Torch `2.15.0.dev20260906+cu130` (`/workspace/venv_bfx9`), `/tmp/attn_gym_up` on the path.
- Model: `kimi_k3_debugmodel` (33 layers, blocks of 12, vocab 163840, dim 1024). The debug config on this tree sets `training.dtype="bfloat16"` (bf16 parameters, gradients and Adam states) and `spmd_backend="partial_dtensor"`; torchtitan's default `training.dtype` is `float32`.
- Cells: `dp1` (1 GPU) and `pp2` (`--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 2`, 1F1B, the cached delta transport `attn_res_cache=True`), 512 tokens per step in two 256-token micro-batches, `--debug.seed 42 --debug.deterministic`, one seed checkpoint (`--checkpoint.create_seed_checkpoint`) loaded by both cells, a warm pass then the measured pass on the same inductor cache (`matrix_scripts/mx3.sh`; scripts `pp_4488style.sh`, `pp_floor_fresh.sh`, `pp_fp32_13.sh` in `matrix_scripts/pp_4488style/`).

```
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --parallelism.data_parallel_shard_degree 1"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 5 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
cell dp1 1
cell pp2 2 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 2
```

## Table 1: the branch's debug config (bf16 parameters and optimizer states), 33 layers

Run `mx3_pprt512_0908_225334`.

| step | loss dp1 | loss pp2 | rel diff | grad norm dp1 | grad norm pp2 | rel diff |
|---|---|---|---|---|---|---|
| 1 | 12.42445 | 12.42445 | 0 | 31.875 | 31.75 | 3.9e-3 |
| 2 | 11.33692 | 11.34505 | 7.2e-4 | 24.375 | 24.375 | 0 |
| 3 | 9.22832 | 9.48903 | 2.8e-2 | 16.5 | 17.875 | 8.3e-2 |
| 4 | 8.59245 | 7.85436 | 8.6e-2 | 14.375 | 14.9375 | 3.9e-2 |
| 5 | 6.51366 | 6.84844 | 5.1e-2 | 9.375 | 10.125 | 8.0e-2 |

Max over 5 steps: loss 8.6e-2, grad norm 8.3e-2. Step 1: loss bitwise; the total norm differs by one bf16 ulp (0.125 at 31.9), the norm being reduced over the two stages (`clip_grad_norm_` sums the stages' squared norms over the pp mesh) in the flavor's bf16.

## Table 2: noise floor, the same dp1 cell on two fresh compile caches

Runs `mx3_pprt512_0908_225334` (cache A) and `mx3_pprt512c_*` (cache B, fresh triton and inductor caches).

| step | loss A | loss B | rel diff | grad norm A | grad norm B | rel diff |
|---|---|---|---|---|---|---|
| 1-5 | 12.42445 / 11.33692 / 9.22832 / 8.59245 / 6.51366 | same | 0 | 31.875 / 24.375 / 16.5 / 14.375 / 9.375 | same | 0 |

Bitwise for 5 steps: `--debug.deterministic` disables the autotuning that moved step 3/10 on earlier trees, so the spread in Table 1 is not compile noise.

## Table 3: float32 parameters and optimizer states (torchtitan's default `training.dtype`), 13 layers

The 33-layer debug model does not fit one 16 GB GPU with float32 parameters, gradients and Adam states (16 bytes per parameter: OOM creating the optimizer states), nor 17 layers (OOM at step 2) nor 13 (OOM at step 3); the five-step table uses a 9-layer run-worktree alias of the same flavor (`debugmodel9`, blocks of 12, so one partial block crosses the boundary), run `mx3_pprt_fp32l9_0908_233404`, same protocol, `--training.dtype float32` (TF32 off in torchtitan, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, flex attention compiled without autotune).

| step | loss dp1 | loss pp2 | rel diff | grad norm dp1 | grad norm pp2 | rel diff |
|---|---|---|---|---|---|---|
| 1 | 12.52069 | 12.52069 | 0 | 14.9514 | 14.9498 | 1.1e-4 |
| 2 | 10.98197 | 10.98491 | 2.7e-4 | 11.1756 | 11.3731 | 1.8e-2 |
| 3 | 9.66179 | 9.67821 | 1.7e-3 | 10.1811 | 10.0973 | 8.2e-3 |
| 4 | 8.40848 | 8.28552 | 1.5e-2 | 9.2338 | 9.2838 | 5.4e-3 |
| 5 | 6.17097 | 6.16612 | 7.9e-4 | 8.1669 | 8.2551 | 1.1e-2 |

Max over 5 steps: loss 1.5e-2, grad norm 1.8e-2. The 13-layer alias gives the same picture for the two steps it survives (step 1: loss bitwise, grad norm 19.478 vs 19.4848, 3.5e-4; step 2: loss 1.3e-3, grad norm 2.0e-2; run `mx3_pprt_fp32l13_0908_232623`).

## Table 4: where the float32 pair separates (step-1 per-parameter gradients, 13 layers)

Per-parameter gradient norm and sha1 before `clip_grad_norm_` (`local_hacks/grad_dump_hack.py` + `grad_hash_hack.py`, compared with `cmp_grad_dumps.py`; dumps `gd_fp32l13/`), dp1 against pp2 (stage 0 = embeddings, vision encoder, layers 0-6; stage 1 = layers 7-12, norm, lm_head, output_res). 432 parameters: 22 bitwise, 410 differ; relative norm difference median 2.2e-4, p90 1.1e-3, max 4.7e-3 (`A_log` of a KDA layer, a 1.6e-4-norm tensor). Both cells are deterministic: pp2 twice on fresh caches is bitwise in all 432 (`pp2b`), dp1 twice on fresh caches is bitwise for 5 steps (Table 2). The difference is ordered by depth in the backward, not by stage:

| parameters | stage | relative norm difference (median over the group) |
|---|---|---|
| lm_head, norm, output_res_norm, output_res_proj | 1 | bitwise |
| layer 12 (last, MLA): moe (10), ffn_res, attention_res_proj, wo, gate, wkv_a, kv_norm | 1 | bitwise |
| layer 12: wq_b, wkv_b (last bits), wq_a 1.2e-7, q_norm 1.4e-6, attention_norm 2.2e-7 | 1 | the first non-identical tensors |
| layer 11 (MLA) | 1 | attention 1.5e-6, moe 4.3e-7, res 3e-5 to 2e-4 |
| layer 10 | 1 | moe 7.8e-6, KDA 6.2e-5, res 3e-4 to 6e-4 |
| layers 9, 8, 7 | 1 | moe 3e-5 to 5e-5, KDA 1e-4, res 3e-4 to 1.4e-3 |
| layers 6, 5, 4 (across the wire) | 0 | moe 1.5e-4 to 1.9e-4, KDA 2e-4, res 5e-4 to 3e-3 |
| layers 3, 2, 1, 0 | 0 | moe 3e-4 to 5e-4, KDA 1e-4 to 1e-3, res 1e-3 to 3.5e-3 |
| tok_embeddings; vision encoder (70) | 0 | 4.1e-4; 4.4e-4 |

Reading: the forward is bitwise (loss, and every gradient that depends only on the forward and the loss gradient: lm_head, the last layer's MoE and output projection). The first difference is a few-ulp one in the query path of the last attention layer's backward (its dO, K, V and output are bitwise, its dQ is not), inside stage 1, in code the pipeline does not touch; walking down the backward it grows by roughly three to five times per layer, and the stage boundary (layer 7 to 6) shows no jump of its own. So with float32 masters the pair is not at the 1e-5 of the DSV3 table: the K3 debug model's backward at random init amplifies an ulp-level difference to 1e-4 to 1e-3 by the bottom layers, and Adam's first update (`lr * sign(g)`) turns the elements whose sign that flips into the percent-level rows of Tables 1 and 3. What the pipeline adds (the wire, the store, the deposits) is below what this profile can see.

Sign census on the same pair (`local_hacks/grad_tensor_dump_hack.py`, `pp_step10_census.py`; dumps `gt_fp32l13/`, cast to bf16 by the hack, which keeps every sign exact): 767.7M elements, 1,008,898 sign flips = 0.131%, 87.8% of them below 1e-2 of their tensor's rms; the implied Adam first-update difference is `2*sqrt(f)` = 7.25% of the update norm, the same order as the 10% the bf16 flavor showed on `pp_review3` (`PP_NUMERICS_FOR_4312_2026-09-05.md`). Per-group relative L2 differences from that dump are not quoted: the bf16 cast rounds every element by up to 2^-8 and dominates them.


## Reading

- The bar (step 1 bitwise loss, gradients within the reduction floor) holds on the client branch: Table 1 step 1, and the per-parameter gradient comparison of `PP_NUMERICS_FOR_4312_2026-09-05.md` (dp1 vs pp2 x vp4: median 2.0e-4, no group stands out, 0.27 percent sign flips on elements below 1e-2 of the tensor rms).
- The percent-level rows after step 1 are not the pipeline's: the fresh-cache floor is zero (Table 2), float32 masters do not remove them (Table 3), and the per-parameter profile (Table 4) puts their origin inside one stage at ulp level with no jump at the boundary; Adam's first update (`lr * sign(g)`) turns the sign flips that amplification produces into percent-level step-2+ differences on this flavor (the mechanism of `k3-step10-spread-is-first-step-sign-flips`). The bf16 flavor adds its own one-ulp total-norm difference at step 1 (Table 1).
- A control that was run and dropped: dp1 with one 512-token micro-batch against dp1 with two of 256 (`mx3_pprt512b_*`) differs at step 1 already (loss 5.2e-4, grad norm 4.3e-2). It is not a reduction-order control: packing 512 tokens into one micro-batch lets tokens after position 256 attend across the boundary that two micro-batches keep hard, so the two runs compute different attention contexts on the same tokens. Not used.
- The single-GPU float32 cell does not fit the 33-layer debug model on a 16 GB GPU (fp32 parameters, gradients and Adam states are 16 bytes per parameter; OOM at optimizer-state creation), nor 17 layers (OOM at step 2) nor 13 (OOM at step 3); 9 layers runs five steps and 13 runs the step-1 probes. The 9/13/17-layer flavors are run-worktree aliases (`debugmodel9/13/17` in the model registry and `kimi_k3_debugmodel9/13/17` in the config registry; `matrix_scripts/local_hacks/debugmodel_layer_aliases_pprt.patch`), never committed. Activation checkpointing is a tyro subcommand on this tree (`activation-checkpoint:full`), not a flag, so it could not be passed through mx3 to shrink the 13-layer cell.
