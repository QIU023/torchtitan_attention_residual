### Summary

Adds context parallelism to the Kimi K3 text decoder. The two attention kinds need different mechanisms, so both are here on disjoint layer kinds: Ulysses on the Gated MLA layers, and the report's KCP on the KDA layers. As far as I can tell this is the first Ulysses in torchtitan; if you would rather it live under distributed/ as a model-agnostic piece, say so and I will move it.

### CP Design: Ulysses on MLA, KCP on KDA

#### Diagram

<img width="1080" height="700" alt="cp_mla_ulysses_flow" src="https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/e3e0aca9dfff9cb868a294e0ada46bada8f4ca74/Raising_PRs/PR_K3_PARALLELISM/cp_mla_ulysses_flow.svg" />

#### Design points: Ulysses on the MLA layers

Ulysses (Jacobs et al., DeepSpeed Ulysses, arXiv:2309.14509). Shapes: T tokens, L = T/cp local tokens, H heads on this rank, G = H/cp, Q/N/V the query, nope-key and value head dims, R the rotary width, W = Q+N+V.

- Entry: the projections run on the local sequence chunk, giving q [L, H, Q], kv [L, H, N+V] and the headless rotary key [L, R]. The layer branches to Ulysses when a CP group was set on it (model.py:131-135; apply_cp_kimi_k3 sets it, parallelize.py:117-133).
- q and kv are packed along the channel dim into one [L, H, W] tensor so a single collective moves both (sharding.py:223).
- All-to-all 1, sequence to heads: [L, H, W] is viewed as [cp, L, G, W] with dim 0 the destination rank, goes through all_to_all_single, and comes back as [T, G, W]: every rank now holds the full sequence for its G heads (sharding.py:156-165, called at 224-227). torch.distributed.nn.functional makes it differentiable; the backward is the transposed all-to-all.
- The packed tensor splits back into q [T, G, Q], k_nope [T, G, N], v [T, G, V] (sharding.py:228-232).
- The rotary key stays out of the all-to-all: it is one vector per token shared by every head, so it is all-gathered along the sequence to [T, R] and expanded onto the G local heads to form k [T, G, N+R] (sharding.py:234-247). Packing the already-expanded key would send the same values once per head and reassemble them against the wrong head subset.
- The attention backend runs unchanged on the full sequence, with a causal mask rebuilt for length T (sharding.py:249-255; full_sequence_causal_mask at 176-196 rejects a folded stream wider than the context window, since a causal-only mask cannot see a document boundary).
- All-to-all 2, heads to sequence: the output [T, G, V] is viewed as [cp, L, G, V] with dim 0 the destination sequence chunk, goes through all_to_all_single, and returns as [L, H, V], the layer's normal layout (sharding.py:166-173, called at 256-259).
- The placement pair S(seq) <-> S(heads) is declared once as a contract (sharding.py:99-109) and the all-to-all takes its dims from it (149-153), so a pair with no implementation raises instead of being ignored. n_heads % cp == 0 is checked at wiring time (parallelize.py:128-132).

#### Design points: KCP on the KDA layers

KCP is not written here: it is fla-core's context parallel for delta-rule recurrences (fla/ops/cp, >= 0.5.1; the README names it KCP after Moonshot's alias). The sequence stays sharded end to end and no rank ever holds it whole.

- Causal conv: a rank needs only the previous rank's last W-1 tokens. causal_conv1d_cp receives them by a point-to-point send/recv and uses them as the conv's initial state; the backward sends the gradient of those tokens back the other way.
- Recurrence: every rank runs the chunk scan from a zero state and emits its end state plus its cumulative decay; one all-gather of these per-head fragments lets each rank compose its true incoming state as a prefix product, and the chunk kernel then runs from it. The backward mirrors it with one all-gather of the state gradients. Per layer that is one all-gather each way and nothing else.
- This branch wires it: one cp_context per forward (kda.py:251-256), the q/k/v convolutions through the halo op (258-275), the kernel call with the context (281-290) and the kernel handing it to chunk_kda (61-95); the import is checked at wiring time so a missing fla-core fails with an actionable message (parallelize.py:138-150).
- Question for the maintainers: torchtitan plans a torch-native KDA. Should that implementation carry KCP as well (the halo conv and the state prefix scan, after fla's), and run under the spmd_types backend by default? If so we would take that on ourselves, and the core change below disappears with it.

#### One core change

KCP cannot be declarative (the fla kernels take raw pointers and never see a DTensor), which is why this model runs on spmd_backend="partial_dtensor", and core's unconditional check rejects context_parallel_degree > 1 on any backend but spmd_types before the model's own wiring runs. So that check moves into a protected method, Decoder.Config._validate_cp_backend, whose default body just calls the existing validate_cp_backend; a model that does not override it runs today's check verbatim, and this model overrides it with its own preconditions. No new config field. The override is tied to the backend, not to how CP is wired: a torch-native KDA that lets this model run under spmd_types deletes it.

### K3 CP runs:

kimi_k3_debugmodel_text at seq 1024 (FlexAttention's BlockMask needs Q_LEN % (cp * 128) == 0), seed 42, --debug.deterministic, one seed checkpoint loaded by every cell; dp2 rows are in because changing the data-parallel degree alone moves the loss more than the sequence split does (step 2: cp2/cp4/cp8 at 2.2e-3/2.6e-3/2.7e-3 from dp1, dp2 at 1.2e-2 from dp1; the mesh cells at 9.9e-3 and 4.0e-3 from dp2):

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.59324 | 6.89766 | 3.33038 |
| cp2 | 2 | 12.59408 | 7.01636 | 3.18600 |
| cp4 | 4 | 12.59255 | 7.18131 | 3.31972 |
| cp8 | 8 | 12.59119 | 7.19044 | 3.29694 |
| dp2 | 2 | 12.59212 | 7.39612 | 3.45591 |
| dp2 x cp2 | 4 | 12.58957 | 7.36327 | 3.48256 |
| dp2 x cp4 | 8 | 12.58948 | 7.43662 | 3.49025 |

Two boundaries raise instead of running: Q_LEN not divisible by cp * 128, and a folded microbatch wider than the context window. Not in this PR: CP inside the vision tower and the report's dynamic CP for large images. Without context_parallel_degree > 1 none of this executes.

### Changed files

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
    torchtitan/models/common/decoder.py  +7/-3  the spmd_types requirement becomes
                                   an overridable method
    tests/
      unit_tests/cpu/test_kimi_k3_cp_contracts.py  +63  the folded-layout contracts
      integration_tests/features.py                 +7  the cp2 cell
    torchtitan_recipes/tests/features.py           +13  its configuration

### CI/CD Coverage

CPU contract tests for the two things that used to fail silently; a cp2 integration cell.

### Numerical Correction run with unmerged upstream grad-norm precision forced to FP32

The same matrix with the grad-norm reduction carried in float32 (https://github.com/pytorch/torchtitan/pull/4135, a separate change not on this branch): context parallelism reaches the same numbers either way, so whatever the sequence split costs, it is not the reduction's grouping.

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.59324 | 6.90509 | 3.29062 |
| cp2 | 2 | 12.59408 | 7.01636 | 3.18646 |
| cp4 | 4 | 12.59255 | 7.18347 | 3.28306 |
| cp8 | 8 | 12.59119 | 7.19044 | 3.29651 |
| dp2 | 2 | 12.59212 | 7.39612 | 3.45638 |
| dp2 x cp2 | 4 | 12.58957 | 7.36461 | 3.42474 |
| dp2 x cp4 | 8 | 12.58948 | 7.43833 | 3.49202 |
