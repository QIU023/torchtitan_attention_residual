Adds pipeline parallelism to the Kimi K3 text decoder. Everything except Block Attention Residuals is mechanical; the residual is not. A block residual is defined over the whole layer stack, so under PP it must travel between stages as a second payload alongside the hidden states, and the final aggregation (output_res_proj, then output_res_norm) must run only on the stage that owns lm_head.

The design, with P the pipeline degree, V the virtual stages per rank, T = P*V stages, and stage S running on rank S mod P under Interleaved1F1B.

- The carrier: a non-head stage returns (hidden, block_residual [tokens, N, D]) and the next stage takes the pair back (model.py:358-404); the head-owning stage alone runs the aggregation (403-407). Without the adapter the whole carrier travels on every hop; that is the fallback transport, and it is what plain 1F1B (V = 1) runs.
- The static layout: layout.py:149-211 simulates one micro-batch's forward in schedule order and tabulates, per stage, the blocks it commits, the blocks its rank's cache already holds, and the delta its P2P must carry (delta = accumulated minus the receiver's cache, 187-194). Sender and receiver compute the same tables, so nothing travels on the wire but the delta.
- Why the delta is bounded: a block committed by stage S_p is fresh on the wire for P-1 hops (S_p+1 .. S_p+P-1); from S_p+P on, every receiving rank already holds it, because its previous virtual stage was S-P. The per-hop payload is bounded by the commits of the last P-1 stages, independent of depth.
- The rank-shared cache: one RankLocalCache per rank (pipeline_adapter.py:140-290), shared by its V virtual stages, keyed (micro-batch, producer stage, commit index).
- Two gradient channels, chosen by whether the producer sits on the consumer's rank. Channel A is PP's own backward P2P: blocks that arrived by recv, and relayed copies of them cached on other ranks, stay autograd-attached to the recv tensor (_finish_forward 744-754, _forward_delta 685-686), so SEND_B carries their gradient hop by hop back to the producing rank, every consumer's contribution merging on that chain.
- Channel B is a rank-local slot bridge for blocks a rank committed itself. The cache holds a DETACHED copy (778-782): a later virtual stage's backward walking into the producer's graph would free it, and the producer's own backward then fails with "backward through the graph a second time". At read time the consumer re-wraps the copy with requires_grad and _LocalCacheCapture (672-684, 365-390), whose backward deposits the gradient in a slot and stops; the grad hook on the producer's attached block (_install_augment_hook 327-362, installed at 761-772) pops the slot and adds it to the incoming gradient during the producer's own backward. No collectives on this path.
- Both channels sum at the producer, so every stage's forward graph is traversed exactly once. The channel-B count is static: for a block from stage S_p with v_p = S_p div P it is V-1-v_p (layout.expected_same_rank_captures 120-145); the hook compares the observed deposits to it and refuses the step on a mismatch (344-356), since a missing capture is a lost gradient no loss curve shows, and the rank's earliest virtual stage asserts every slot drained at micro-batch end (868-890).
- The carry is a config field, attn_res_cache, on by default under PP; a launcher exporting an environment variable non-uniformly would give ranks different topologies and hang in a collective with nothing pointing at the cause.

Splitting the model in two by hand and running the halves in sequence -- no schedule, no loss, no microbatches -- reproduces the unsplit forward at max_abs 0.000e+00.

kimi_k3_debugmodel_text_32l, seed 42, --debug.deterministic, every cell loading the same seed checkpoint; each cell runs twice and the first run is discarded, because a cold compile cache moves this model's step-1 loss by 6.8e-3 (12.59459 against 12.58783, both reproducible). Step 1 is bit-identical to dp1 in all nine cells, two to thirty-two stages, two to eight ranks; the transport is the delta one wherever the schedule can carry it and falls back on plain 1F1B, where a rank holds one stage:

| cell | stages | world | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | 12.48548 | 7.92534 | 3.41439 |
| pp2 | 2 | 2 | fallback | 12.48548 | 7.91227 | 3.35806 |
| pp4 | 4 | 4 | fallback | 12.48548 | 7.91227 | 3.35881 |
| pp8 | 8 | 8 | fallback | 12.48548 | 7.90930 | 3.40345 |
| pp2 x vp2 | 4 | 2 | delta | 12.48548 | 7.89923 | 3.42837 |
| pp2 x vp4 | 8 | 2 | delta | 12.48548 | 7.94984 | 3.39687 |
| pp4 x vp2 | 8 | 4 | delta | 12.48548 | 7.93775 | 3.28493 |
| pp4 x vp4 | 16 | 4 | delta | 12.48548 | 7.89573 | 3.33794 |
| pp8 x vp2 | 16 | 8 | delta | 12.48548 | 7.93965 | 3.25196 |
| pp8 x vp4 | 32 | 8 | delta | 12.48548 | 7.91517 | 3.38148 |

The same six virtual-stage cells with the transport turned off, against the rows above: same forward, different gradients, the same blocks routed a different way and summed in a different order.

| cell | step 1 | step 2 | step 3 | step 10 |
|---|---|---|---|---|
| pp2 x vp2 | identical | 3.5e-4 | 1.6e-3 | 2.1e-2 |
| pp2 x vp4 | identical | 3.5e-4 | 5.1e-3 | 1.7e-3 |
| pp4 x vp2 | identical | 1.5e-4 | 3.2e-3 | 2.2e-2 |
| pp4 x vp4 | identical | 4.9e-4 | 1.7e-3 | 1.9e-2 |
| pp8 x vp2 | identical | 5.2e-4 | 3.8e-3 | 4.5e-2 |
| pp8 x vp4 | identical | 6.4e-4 | 7.4e-4 | 6.6e-3 |

Peak memory per rank, same topology and schedule, transport against fallback; the six ranks that hold little are unchanged, the saving is on the two that would otherwise accumulate the stack (pp8 x vp2 is the same shape, 7.92 down to 6.03):

| pp8 x vp4 | per-rank peak (GiB) | max | spread |
|---|---|---|---|
| delta | 2.62 x6, 6.57, 6.60 | 6.60 | 3.98 |
| fallback | 2.66 x6, 7.83, 8.50 | 8.50 | 5.84 |

The same matrix with the grad-norm reduction carried in float32 (https://github.com/pytorch/torchtitan/pull/4135, a separate change not on this branch): the six virtual-stage cells with the transport off collapse to one value (7.87346, all six), and with it on they stay apart, because the delta a hop carries depends on the cut and summing it in a different order is arithmetic that patch does not touch.

| cell | stages | world | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | 12.48548 | 7.90621 | 3.40567 |
| pp2 | 2 | 2 | fallback | 12.48548 | 7.87346 | 3.43178 |
| pp4 | 4 | 4 | fallback | 12.48548 | 7.87346 | 3.43178 |
| pp8 | 8 | 8 | fallback | 12.48548 | 7.91590 | 3.41976 |
| pp2 x vp2 | 4 | 2 | delta | 12.48548 | 7.90505 | 3.34575 |
| pp2 x vp4 | 8 | 2 | delta | 12.48548 | 7.94826 | 3.32969 |
| pp4 x vp2 | 8 | 4 | delta | 12.48548 | 7.95831 | 3.38758 |
| pp4 x vp4 | 16 | 4 | delta | 12.48548 | 7.90776 | 3.31534 |
| pp8 x vp2 | 16 | 8 | delta | 12.48548 | 7.92740 | 3.31839 |
| pp8 x vp4 | 32 | 8 | delta | 12.48548 | 7.91857 | 3.37609 |

Not in this PR: the vision tower (its stage assignment and DEP). Without pipeline_parallel_degree > 1 none of this executes.

Files:

    torchtitan/models/kimi_k3/
      pipeline_adapter.py   +1228  the pipelining_fn: FQN split, the block-residual
                                   carry across stages, and the rank-shared stack
      layout.py              +293  offline algebra over (pp, vp, num_blocks,
                                   n_layers, layers_per_block): which blocks each
                                   stage commits, which subset its P2P ships
      __init__.py           +47/-5 registers pipelining_fn and the 32-layer text
                                   flavor; zero-init on the AttnRes projections
      model.py              +26/-3 returns (hidden, block_residual) off a non-head
                                   stage, takes the pair back on the next, guards
                                   the head-only aggregation; the attn_res_cache
                                   field that selects the transport
      config_registry.py       +35 the 32-layer trainer flavor, and its _naive
                                   twin with the transport off
      parallelize.py         +2/-3 pipeline parallel off the unsupported list
    tests/
      unit_tests/cpu/test_kimi_k3_pp_fqn_injection.py  +105  the FQN split, on CPU
      integration_tests/features.py                     +14  pp2 and pp8 x vp4 cells
    torchtitan_recipes/tests/features.py                +33  their configurations

Tested: a CPU unit test for the FQN split; a pp2 integration cell, and a pp8 x vp4 one on the 32-layer flavor -- one layer per stage over 32 stages, so the residual crosses every boundary the schedule has.
