Enables expert parallelism for Kimi K3 on the existing all-to-all token dispatcher. Three pieces: set_moe_sharding_config declares the routed-expert layout (w1_EFD S(1), w2_EDF S(2), w3_EFD S(1)), with set_decoder_sharding_config above it so the activations reaching the MoE boundary are DTensors that can be redistributed onto the expert mesh; parallelize builds the expert-data-parallel mesh that excludes the expert axis and hands it to apply_fsdp_to_decoder with ep_degree -- the shape deepseek_v3 already resolves; and expert parallel comes off the unsupported list.

EP shards experts inside the data axis, so each cell is compared against the pure-data-parallel run at the same world size. kimi_k3_debugmodel_text, seed 42, --debug.deterministic, one seed checkpoint loaded by every cell:

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.58955 | 7.55365 | 3.46009 |
| dp2 | 2 | 12.57725 | 7.57439 | 3.19128 |
| ep2 x fsdp2 | 2 | 12.57718 | 7.64509 | 3.20056 |
| dp4 | 4 | 12.59366 | 7.20509 | 3.30981 |
| ep4 x fsdp4 | 4 | 12.59386 | 7.25484 | 3.31417 |
| dp8 | 8 | 12.58336 | 7.32668 | 3.36729 |
| ep8 x fsdp8 | 8 | 12.58286 | 7.40832 | 3.44799 |

At step 1 each EP cell is within 5.6e-6, 1.6e-5 and 4.0e-5 relative of the
pure-data-parallel run at its own degree. At step 2 they are 2.9e-4, 1.2e-3 and
1.1e-3, against 3.7e-3, 3.5e-3 and 5.2e-3 for the dp2, dp4 and dp8 rows measured
the same way against dp1 -- sharding the experts moves the loss less than
changing the data-parallel degree does.

MoonEP is not here: the report's balanced EP with online redundant-expert planning needs its own dispatcher and backend, and this is the plain all-to-all path. comm_backend is pinned to "standard" so no run can silently believe it is on MoonEP. Without expert_parallel_degree > 1 none of this executes.

A second measurement with one patch on top: the grad-norm reduction carried in
float32 rather than in the gradients' dtype. That patch is a separate upstream change, still open at https://github.com/pytorch/torchtitan/pull/4135, and is not on this branch.

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.58955 | 7.54919 | 3.50680 |
| dp2 | 2 | 12.57725 | 7.57439 | 3.19060 |
| ep2 x fsdp2 | 2 | 12.57718 | 7.64509 | 3.20221 |
| dp4 | 4 | 12.59366 | 7.20509 | 3.31005 |
| ep4 x fsdp4 | 4 | 12.59386 | 7.25484 | 3.30867 |
| dp8 | 8 | 12.58336 | 7.32668 | 3.36718 |
| ep8 x fsdp8 | 8 | 12.58286 | 7.42252 | 3.43101 |

Only ep8 moves: 7.40832 to 7.42252 at step 3, while ep2, ep4 and all four
pure-data-parallel rows are unchanged to every printed digit. The reduction is
taken over an expert group and a non-expert group separately, so the total
depends on that split; at two and four experts per group the two norms are
close enough that bf16 rounds them the same way, and at eight they are not.
Against its own-degree baseline with the patch applied, ep2 is 2.8e-4 at step 2
and ep8 is 5.1e-4.

Files:

    torchtitan/models/kimi_k3/
      model.py                 +31  _set_sharding_config: the routed-expert layout
                                   and the decoder-level distribution above it
      __init__.py           +19/-2 core's dispatcher factory, pinned to the
                                   standard all-to-all; the text flavor's spec
      config_registry.py       +15  the text trainer flavor
      parallelize.py         +14/-2 the efsdp mesh, ep_degree through to
                                   apply_fsdp_to_decoder, and expert parallel off
                                   the unsupported list
    tests/
      unit_tests/cpu/test_kimi_k3_ep_sharding.py  +51  the declaration, on CPU
      integration_tests/features.py                +7  the ep2 cell
    torchtitan_recipes/tests/features.py          +13  its configuration

Tested: a CPU test on the declaration (plain DP declares nothing; EP shards the routed experts on the expert axis); an ep2 integration cell.
