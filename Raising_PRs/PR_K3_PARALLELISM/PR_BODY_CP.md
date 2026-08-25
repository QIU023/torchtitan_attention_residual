Adds context parallelism to the Kimi K3 text decoder. The two attention kinds need different mechanisms, so both are here: Ulysses on the Gated MLA layers, and the report's KCP on the KDA layers -- a conv halo exchange plus a prefix scan over the recurrent state, nothing softmax-shaped. As far as I can tell this is the first Ulysses in torchtitan; if you would rather it live under distributed/ as a model-agnostic piece, say so and I will move it.

Ulysses is one all-to-all trading the sharded axis from sequence to heads, the unchanged attention backend, and a second all-to-all trading back. The rotary key slice stays outside the exchange: it is headless -- one vector per token shared by every head -- so it is all-gathered along the sequence and expanded onto local heads afterwards; packing the already-expanded key sends the same values once per head and reassembles them against the wrong head subset.

KCP cannot be declarative: the fla kernels take raw pointers and never dispatch through DTensor, so no ShardingConfig reaches them. Hence one core change: the spmd_types requirement moves into a protected method, Decoder.Config._validate_cp_backend, and a model whose CP is not ShardingConfig-driven overrides it with its own preconditions. The method's default body just calls the existing validate_cp_backend, so it is only the override point: a model that does not override it runs today's check verbatim. No new config field; declarative models are unchanged.

CP resplits the sequence, so unlike PP and EP it is not expected to be bit-identical to dp1. One seed checkpoint is loaded by every cell; dp2 is in the table because changing the data-parallel degree alone moves the loss more than any of these axes do, so a dp2-mesh cell measured against dp1 would mostly be measuring that. kimi_k3_debugmodel_text at seq 1024 -- FlexAttention's BlockMask needs Q_LEN % (cp * 128) == 0, which is what sets the length -- seed 42, --debug.deterministic:

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.59324 | 6.89766 | 3.33038 |
| cp2 | 2 | 12.59408 | 7.01636 | 3.18600 |
| cp4 | 4 | 12.59255 | 7.18131 | 3.31972 |
| cp8 | 8 | 12.59119 | 7.19044 | 3.29694 |
| dp2 | 2 | 12.59212 | 7.39612 | 3.45591 |
| dp2 x cp2 | 4 | 12.58957 | 7.36327 | 3.48256 |
| dp2 x cp4 | 8 | 12.58948 | 7.43662 | 3.49025 |

Step 1 is within 1.6e-4 relative of dp1 across every cell. At step 2, the three
single-axis cells sit at 2.2e-3, 2.6e-3 and 2.7e-3 against dp1, while dp2 --
which enables no parallelism axis at all -- sits at 1.2e-2: the sequence split
moves the loss less than changing the data-parallel degree does. The two mesh
cells are 9.9e-3 and 4.0e-3 against dp2.

Two boundaries raise instead of running: Q_LEN not divisible by cp * 128, and a folded microbatch wider than the context window, since the CP mask rebuild is causal-only and cannot represent a document boundary.

Not in this PR: CP inside the vision tower and the report's dynamic CP for large images -- next, on the multimodal path. Without context_parallel_degree > 1 none of this executes.

Files:

    torchtitan/models/kimi_k3/
      sharding.py            +259  the two CP contracts, the head/sequence
                                   all-to-all, MLA's Ulysses body and the
                                   causal-mask rebuild it needs
      kda.py                 +120  KCP on the KDA layers: conv halo exchange and
                                   the prefix scan over the recurrent state
      model.py              +55/-14 MLA branches to Ulysses when a CP group is set;
                                   overrides _validate_cp_backend
      parallelize.py         +71/-5 apply_cp_kimi_k3: wires the group onto both
                                   layer kinds, checks head divisibility, and fails
                                   at wiring time if fla's CP ops are missing
      __init__.py              +13  the text flavor's model spec
      config_registry.py       +13  the text trainer flavor
    torchtitan/models/common/decoder.py  +9/-3  the spmd_types requirement becomes
                                   an overridable method
    tests/
      unit_tests/cpu/test_kimi_k3_cp_contracts.py  +63  the folded-layout contracts
      integration_tests/features.py                 +7  the cp2 cell
    torchtitan_recipes/tests/features.py           +13  its configuration

Tested: CPU contract tests for the two things that used to fail silently; a cp2 integration cell.
