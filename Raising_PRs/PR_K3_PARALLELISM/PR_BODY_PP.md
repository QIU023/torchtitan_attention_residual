### Summary

Adds pipeline parallelism to the Kimi K3 text decoder. Everything except Block Attention Residuals is mechanical; the residual is not. A block residual is defined over the whole layer stack, so under PP it must travel between stages as a second payload alongside the hidden states, and the final aggregation (`output_res_proj`, then `output_res_norm`) must run only on the stage that owns `lm_head`.


### PP Virtual Stage Cache Adapter Design for Attention Residual

#### Diagram

<img width="1080" height="540" alt="pp_dual_gradient_bridge" src="https://github.com/user-attachments/assets/8a5674a0-575f-4ab0-80ab-b22ace2a67b4" />

#### Design points

with $`P`$ the pipeline degree, $`V`$ the virtual stages per rank, $`T = P \cdot V`$ stages, and stage $`S`$ running on rank $`S \bmod P`$ under `Interleaved1F1B`.

- The carrier
  - A non-head stage returns (*hidden*, *block_residual*) with *block_residual* $`[\mathrm{tokens}, N, D]`$ and the next stage takes the pair back (`model.py:358-404`); the head-owning stage alone runs the aggregation (`model.py:403-407`).
  - Without the adapter the whole carrier travels on every hop: that is the fallback transport, and it is what plain `1F1B` ($`V = 1`$) runs.
  - The transport is a config field, `attn_res_cache`, on by default under PP; a launcher exporting an environment variable non-uniformly would give ranks different topologies and hang in a collective with nothing pointing at the cause.
- The static layout
  - `layout.py:149-211` (`BlockLayoutTables._build`) simulates one micro-batch's forward in schedule order and tabulates, per stage, the blocks it commits, the blocks its rank's cache already holds, and the delta its P2P must carry (delta = accumulated minus the receiver's cache, `layout.py:187-194`). Sender and receiver compute the same tables, so nothing travels on the wire but the delta.
  - Why the delta is bounded: a block committed by stage $`S_p`$ is fresh on the wire for $`P-1`$ hops ($`S_p+1, \ldots, S_p+P-1`$); from $`S_p+P`$ on, every receiving rank already holds it, because its previous virtual stage was $`S-P`$. The per-hop payload is bounded by the commits of the last $`P-1`$ stages, independent of depth.
- The rank-shared cache
  - One `RankLocalCache` per rank (`pipeline_adapter.py:140-290`), shared by its $`V`$ virtual stages, keyed (micro-batch, producer stage, commit index).
  - Blocks that arrived by recv are cached attached. Blocks the rank committed itself are cached as a DETACHED copy (`pipeline_adapter.py:778-782` in `_finish_forward`): a later virtual stage's backward walking into the producer's graph would free it, and the producer's own backward then fails with "backward through the graph a second time".
- The two gradient channels, chosen by whether the producer sits on the consumer's rank
  - Channel A, PP's own backward P2P: blocks that arrived by recv, and relayed copies of them cached on other ranks, stay autograd-attached to the recv tensor (`pipeline_adapter.py:744-754` in `_finish_forward`, `pipeline_adapter.py:685-686` in `_forward_delta`), so `SEND_B` carries their gradient hop by hop back to the producing rank, every consumer's contribution merging on that chain.
  - Channel B, the rank-local slot bridge: at read time the consumer re-wraps the detached copy with `requires_grad` and `_LocalCacheCapture` (`pipeline_adapter.py:672-684` in `_forward_delta`; the Function at `pipeline_adapter.py:365-390`), whose backward deposits the gradient in a slot and stops; the grad hook on the producer's attached block (`_install_augment_hook`, `pipeline_adapter.py:327-362`; installed at `pipeline_adapter.py:761-772` in `_finish_forward`) pops the slot and adds it to the incoming gradient during the producer's own backward. No collectives on this path.
  - Both channels sum at the producer, so every stage's forward graph is traversed exactly once.
- The self-check
  - The channel-B count is static: for a block from stage $`S_p`$ with $`v_p = \lfloor S_p / P \rfloor`$ it is $`V-1-v_p`$ (`layout.py:120-145`, `expected_same_rank_captures`); the hook compares the observed deposits to it and refuses the step on a mismatch (`pipeline_adapter.py:344-356` in `_install_augment_hook`), since a missing capture is a lost gradient no loss curve shows.
  - The rank's earliest virtual stage asserts every slot drained at micro-batch end (`pipeline_adapter.py:868-890`, `on_microbatch_end`).

### K3 PP with VP runs:

Splitting the model in two by hand and running the halves in sequence -- no schedule, no loss, no microbatches -- reproduces the unsplit forward at max_abs 0.000e+00.

To reproduce, from the torchtitan checkout root on this branch, 8 GPUs. Every cell loads the same seed checkpoint; run each cell twice and read the second run (a cold compile cache moves step 1). The runner we used, with the seed-load assertion and a disk gate, is https://github.com/QIU023/torchtitan_attention_residual/blob/611385d4e123d4d0527c6d08b06f8d701bb63e21/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh.

```sh
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_32l --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --parallelism.data_parallel_shard_degree 1"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
P="--parallelism.pipeline_parallel_degree"; L="--parallelism.pipeline-parallel-layers-per-stage"
MB="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
cell dp1 1
cell pp2 2 $P 2 $MB;  cell pp4 4 $P 4 $MB;  cell pp8 8 $P 8 $MB
cell pp2_vp2 2 $P 2 $L 8 $IL;  cell pp2_vp4 2 $P 2 $L 4 $IL
cell pp4_vp2 4 $P 4 $L 4 $IL;  cell pp4_vp4 4 $P 4 $L 2 $IL
cell pp8_vp2 8 $P 8 $L 2 $IL;  cell pp8_vp4 8 $P 8 $L 1 $IL
# the transport-off rows: the same cells with --config kimi_k3_debugmodel_32l_naive
```

`kimi_k3_debugmodel_32l`, seed 42, `--debug.deterministic`, every cell loading the same seed checkpoint; each cell runs twice and the first run is discarded, because a cold compile cache moves this model's step-1 loss. Step 1 is bit-identical to dp1 in all nine cells, two to thirty-two stages, two to eight ranks; the transport is the delta one wherever the schedule can carry it and falls back on plain `1F1B`, where a rank holds one stage:

| cell | stages | world | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | 12.47877 | 7.37540 | 3.41780 |
| pp2 | 2 | 2 | fallback | 12.47877 | 7.31716 | 3.42172 |
| pp4 | 4 | 4 | fallback | 12.47877 | 7.31098 | 3.50222 |
| pp8 | 8 | 8 | fallback | 12.47877 | 7.27499 | 3.63363 |
| pp2 x vp2 | 4 | 2 | delta | 12.47877 | 7.28330 | 3.49022 |
| pp2 x vp4 | 8 | 2 | delta | 12.47877 | 7.20360 | 3.58730 |
| pp4 x vp2 | 8 | 4 | delta | 12.47877 | 7.25567 | 3.43437 |
| pp4 x vp4 | 16 | 4 | delta | 12.47877 | 7.29147 | 3.52714 |
| pp8 x vp2 | 16 | 8 | delta | 12.47877 | 7.26481 | 3.42243 |
| pp8 x vp4 | 32 | 8 | delta | 12.47877 | 7.28763 | 3.46361 |

Three of the virtual-stage cells re-run with the transport turned off, against the rows above: same forward, different gradients, the same blocks routed a different way and summed in a different order.

| cell | step 1 | step 2 |
|---|---|---|
| pp2 x vp2 | identical | 1.0e-3 |
| pp4 x vp4 | identical | 4.0e-4 |
| pp8 x vp4 | identical | 1.6e-3 |

Peak memory per rank, same topology and schedule, transport against fallback; the six ranks that hold little are unchanged, the max comes down from 8.50 to 7.53 GiB and the spread from 5.83 to 4.90:

| pp8 x vp4 | per-rank peak (GiB) | max | spread |
|---|---|---|---|
| delta | 2.63 x6, 6.61, 7.53 | 7.53 | 4.90 |
| fallback | 2.67 x6, 8.50, 7.09 | 8.50 | 5.83 |

Not in this PR: the vision tower (its stage assignment and DEP). Without `pipeline_parallel_degree > 1` none of this executes.

### Cache offload (review branch, 2026-09-02)

`attn_res_cache_offload`, off by default: own-rank cached commits park on pinned host memory between the producing stage's forward and their same-rank consumers' forwards. Only own commits move -- their consumer linkage routes through the Capture/Augment slot bridge, never through shared storage, so the host round-trip is value-identical; relayed blocks stay attached on device for SEND_B. The D2H copy is async on the current stream and the H2D in `get_blocks` runs on the same stream, so stream order serializes them.

Value-identical is the claim and the pp2 x vp2 cell prints it to every digit; at debug scale the parked blocks are about a megabyte each, so peak memory does not move here -- the saving scales with microbatch tokens x hidden x blocks in flight.

| cell | offload | step 1 | step 3 | step 10 | peak memory |
|---|---|---|---|---|---|
| pp2 x vp2 | off | 12.49999 | 6.89362 | 3.28050 | 10.43 GiB |
| pp2 x vp2 | on | 12.49999 | 6.89362 | 3.28050 | 10.42 GiB |

    torchtitan/models/kimi_k3/
      model.py                 +6/-0   the attn_res_cache_offload field
      pipeline_adapter.py      +43/-3  RankLocalCache parks own commits on pinned host memory

### HF import under pipeline parallelism (review branch, 2026-09-02)

`KimiK3StateDictAdapter.to_hf` synthesizes the released checkpoint's unused layer-0 attention-residual placeholders from layer 1's tensors; a pipeline stage's state dict holds its own layers only, so the synthesis now runs where layer 0 lives and the layer-1 templates are present, and is skipped elsewhere. Found by loading the HF debug checkpoint into a two-stage pipeline: the second stage raised `KeyError: 'language_model.model.layers.1.self_attention_res_norm.weight'`.

    torchtitan/models/kimi_k3/
      state_dict_adapter.py    +12/-6  stage-aware placeholder synthesis

### Changed files

    torchtitan/models/kimi_k3/
      pipeline_adapter.py   +1228  the pipelining_fn: FQN split, the block-residual
                                   carry across stages, and the rank-shared stack
      layout.py              +293  offline algebra over (pp, vp, num_blocks,
                                   n_layers, layers_per_block): which blocks each
                                   stage commits, which subset its P2P ships
      __init__.py           +33/-5 registers pipelining_fn and the 32-layer
                                   flavor; zero-init on the AttnRes projections
      model.py              +26/-3 returns (hidden, block_residual) off a non-head
                                   stage, takes the pair back on the next, guards
                                   the head-only aggregation; the attn_res_cache
                                   field that selects the transport
      config_registry.py       +23 the 32-layer trainer flavor, and its _naive
                                   twin with the transport off
      parallelize.py         +2/-3 pipeline parallel off the unsupported list
    tests/
      unit_tests/cpu/test_kimi_k3_pp_fqn_injection.py  +105  the FQN split, on CPU
      integration_tests/features.py                     +14  pp2 and pp8 x vp4 cells
    torchtitan_recipes/tests/features.py                +33  their configurations

### CI/CD Coverage

CPU unit test for the FQN split; a pp2 integration cell, and a pp8 x vp4 one on the 32-layer flavor -- one layer per stage over 32 stages, so the residual crosses every boundary the schedule has.


### Numerical Correction run with unmerged upstream grad-norm precision forced to FP32

The same matrix with the grad-norm reduction carried in float32 (https://github.com/pytorch/torchtitan/pull/4135, a separate change not on this branch): with the transport off, pp4 x vp4 and pp8 x vp4 collapse onto one curve (7.27054 at step 3 and 3.33147 at step 10, exactly), and with the transport on the same two cells stay apart, because the delta a hop carries depends on the cut and summing it in a different order is arithmetic that patch does not touch.

| cell | stages | world | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | 12.47877 | 7.22437 | 3.62374 |
| pp2 | 2 | 2 | fallback | 12.47877 | 7.32454 | 3.61987 |
| pp4 | 4 | 4 | fallback | 12.47877 | 7.35269 | 3.42883 |
| pp8 | 8 | 8 | fallback | 12.47877 | 7.22659 | 3.42034 |
| pp2 x vp2 | 4 | 2 | delta | 12.47877 | 7.28511 | 3.44289 |
| pp2 x vp4 | 8 | 2 | delta | 12.47877 | 7.28217 | 3.35374 |
| pp4 x vp2 | 8 | 4 | delta | 12.47877 | 7.27053 | 3.42265 |
| pp4 x vp4 | 16 | 4 | delta | 12.47877 | 7.24882 | 3.38527 |
| pp8 x vp2 | 16 | 8 | delta | 12.47877 | 7.27183 | 3.31784 |
| pp8 x vp4 | 32 | 8 | delta | 12.47877 | 7.26931 | 3.31291 |
