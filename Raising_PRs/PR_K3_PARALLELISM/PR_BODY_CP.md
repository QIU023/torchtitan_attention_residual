### Summary

Adds context parallelism to the Kimi K3 text decoder. The two attention kinds need different mechanisms, so both are here on disjoint layer kinds: Ulysses on the Gated MLA layers, and the report's KCP on the KDA layers. As far as I can tell this is the first Ulysses in torchtitan; if you would rather it live under `distributed/` as a model-agnostic piece, say so and I will move it.

### CP Design: Ulysses on MLA, KCP on KDA

#### Diagram

<img width="1080" height="440" alt="cp_mla_ulysses_flow" src="https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/26bf162c61caa4946f9ef251a47304a2e6ceabb6/Raising_PRs/PR_K3_PARALLELISM/cp_mla_ulysses_flow.svg" />

#### Design points: Ulysses on the MLA layers

Ulysses (Jacobs et al., DeepSpeed Ulysses, arXiv:2309.14509). Shapes: $`T`$ tokens, $`L = T/cp`$ local tokens, $`H`$ heads on this rank, $`G = H/cp`$, $`Q/N/V`$ the query, nope-key and value head dims, $`R`$ the rotary width, $`W = Q+N+V`$.

- Before the first all-to-all: sequence-sharded, all heads local
  - The projections run on this rank's chunk of $`L`$ tokens and give *q* $`[L, H, Q]`$, *kv* $`[L, H, N+V]`$ and the headless rotary key *k_rope* $`[L, R]`$. The layer takes this path when a CP group was set on it (`model.py:132-136` in `KimiMLAAttention.forward`; `apply_cp_kimi_k3` sets it, `parallelize.py:117-133`).
  - *q* and *kv* are packed along the channel dim into one $`[L, H, W]`$ tensor so a single collective moves both (`sharding.py:223` in `mla_ulysses_attention`).
- All-to-all 1: sequence-sharded to head-sharded
  - $`[L, H, W]`$ is viewed as $`[cp, L, G, W]`$ with dim 0 the destination rank and goes through `all_to_all_single`; it comes back as $`[T, G, W]`$, the full sequence for this rank's $`G`$ heads (`sharding.py:156-165` in `cp_all_to_all_headseq`, called at `sharding.py:224-227`).
  - `torch.distributed.nn.functional` makes it differentiable: the backward is the transposed all-to-all.
- Inside the rank: head-sharded, full-sequence MLA
  - The packed tensor splits back into *q* $`[T, G, Q]`$, *k_nope* $`[T, G, N]`$, *v* $`[T, G, V]`$ (`sharding.py:228-232`).
  - The rotary key stays out of the all-to-all: it is one vector per token shared by every head, so it is all-gathered along the sequence to $`[T, R]`$ and expanded onto the $`G`$ local heads to form *k* $`[T, G, N+R]`$ (`sharding.py:234-247`). Packing the already-expanded key would send the same values once per head and reassemble them against the wrong head subset.
  - The attention backend runs unchanged on the full sequence, with a causal mask rebuilt for length $`T`$ (`sharding.py:249-255`; `full_sequence_causal_mask`, `sharding.py:176-196`, rejects a folded stream wider than the context window, since a causal-only mask cannot see a document boundary).
- All-to-all 2: head-sharded back to sequence-sharded
  - The output $`[T, G, V]`$ is viewed as $`[cp, L, G, V]`$ with dim 0 the destination sequence chunk, goes through `all_to_all_single`, and returns as $`[L, H, V]`$ (`sharding.py:166-173` in `cp_all_to_all_headseq`, called at `sharding.py:256-259`).
- After the second all-to-all: sequence-sharded again
  - $`[L, H, V]`$ is the layer's normal layout; the output projection and the gate run on it exactly as without CP (`model.py:145-146`).
  - The placement pair `S(seq) <-> S(heads)` is declared once as a contract (`sharding.py:99-109`, `ULYSSES`) and the all-to-all takes its dims from it (`sharding.py:149-153`), so a pair with no implementation raises instead of being ignored. `n_heads % cp == 0` is checked at wiring time (`parallelize.py:128-132` in `apply_cp_kimi_k3`).

#### Design points: KCP on the KDA layers

KCP is not written here: it is fla-core's context parallel for delta-rule recurrences (`fla/ops/cp`, >= 0.5.1; the README names it KCP after Moonshot's alias). The sequence stays sharded end to end and no rank ever holds it whole.

- Causal conv: a rank needs only the previous rank's last $`W-1`$ tokens. `causal_conv1d_cp` receives them by a point-to-point send/recv and uses them as the conv's initial state; the backward sends the gradient of those tokens back the other way.
- Recurrence: every rank runs the chunk scan from a zero state and emits its end state plus its cumulative decay; one all-gather of these per-head fragments lets each rank compose its true incoming state as a prefix product, and the chunk kernel then runs from it. The backward mirrors it with one all-gather of the state gradients. Per layer that is one all-gather each way and nothing else.
- This branch wires it: one `cp_context` per forward (`kda.py:251-256` in `_forward_kcp`), the q/k/v convolutions through the halo op (`kda.py:258-275`), the kernel call with the context (`kda.py:281-290`) and the kernel handing it to `chunk_kda` (`kda.py:61-95` in `KimiKDAKernel.forward`); the import is checked at wiring time so a missing fla-core fails with an actionable message (`parallelize.py:138-150` in `apply_cp_kimi_k3`).
- Question for the maintainers: torchtitan plans a torch-native KDA. Should that implementation carry KCP as well (the halo conv and the state prefix scan, after fla's), and run under the `spmd_types` backend by default? If so we would take that on ourselves, and the core change below disappears with it.

#### Design points: the multimodal splice under CP

- Upstream ships one flavor and it is multimodal. `prepare_context_parallel_input` shards tokens, labels and positions along the sequence but leaves `pixel_values` whole, so every rank encodes every image while holding only a slice of the placeholders, and `get_vision_positions` refuses a slice that splits a visual item or holds none.
- Each rank now scatters the feature slice its own placeholders correspond to (`model.py:379-393`): CP shards are contiguous and equal (the config already rejects a load balancer under CP), so a rank's slice starts after the placeholders the lower ranks hold, which one all-reduce establishes (`model.py:419`).
- The rows a rank does not consume stay in the graph through `add_zero_valued_dependency` (`torchtitan/distributed/fsdp.py:168-189`), so FSDP2 issues the tower's reduce-scatter on every rank. The encoder still runs redundantly on every CP rank; splitting the tower is the later DEP and dynamic-CP work.

#### Two core changes

KCP cannot be declarative (the fla kernels take raw pointers and never see a DTensor), which is why this model runs on `spmd_backend="partial_dtensor"`, and core's unconditional check rejects `context_parallel_degree > 1` on any backend but `spmd_types` before the model's own wiring runs. So that check moves into a protected method, `Decoder.Config._validate_cp_backend`, whose default body just calls the existing `validate_cp_backend`; a model that does not override it runs today's check verbatim, and this model overrides it with its own preconditions. No new config field. The override is tied to the backend, not to how CP is wired: a torch-native KDA that lets this model run under `spmd_types` deletes it.

The second is `add_zero_valued_dependency` in `torchtitan/distributed/fsdp.py:168-189`: a helper that keeps a partly consumed FSDP module in the autograd graph by adding a zero-scaled sum of its unused output, so the collectives FSDP2 hangs on that output are issued by every rank. It is not K3-specific; any model whose ranks consume different parts of an FSDP module's output needs the same edge.

### K3 CP runs:

To reproduce, from the torchtitan checkout root on this branch, 8 GPUs. Every cell loads the same seed checkpoint; run each cell twice and read the second run (a cold compile cache moves step 1). The runner we used, with the seed-load assertion and a disk gate, is https://github.com/QIU023/torchtitan_attention_residual/blob/611385d4e123d4d0527c6d08b06f8d701bb63e21/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh.

```sh
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 16384 --training.num-tokens-per-microbatch-per-dp-rank 1024 --training.max-context-length 1024 --checkpoint.enable"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"; NB="--parallelism.context_parallel_load_balancer None"
cell dp1 1 $D 1
cell cp2 2 $D 1 $C 2 $NB;  cell cp4 4 $D 1 $C 4 $NB;  cell cp8 8 $D 1 $C 8 $NB
cell dp2 2 $D 2;  cell dp2_cp2 4 $D 2 $C 2 $NB;  cell dp2_cp4 8 $D 2 $C 4 $NB
```

`kimi_k3_debugmodel` at seq 1024 (`FlexAttention`'s `BlockMask` needs `Q_LEN % (cp * 128) == 0`), seed 42, `--debug.deterministic`, one seed checkpoint loaded by every cell; dp2 rows are in because changing the data-parallel degree alone moves the loss more than the sequence split does. Measured after the packed-document mask fix from review (below): this dataset packs two documents per 1024-token stream, so the earlier causal-only CP rebuild let the second document attend the first. Relative to the pre-fix table the non-CP rows (dp1, dp2) are bitwise unchanged and every CP row moves -- the fixed cells now compute the same masking dp1 does; cp2 reproduces to every printed digit on a same-seed re-run:

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.60544 | 7.30226 | 3.22742 |
| cp2 | 2 | 12.53996 | 7.04577 | 3.50511 |
| cp4 | 4 | 12.52432 | 6.89080 | 3.49263 |
| cp8 | 8 | 12.53711 | 7.52113 | 3.49416 |
| dp2 | 2 | 12.58193 | 7.44923 | 3.32128 |
| dp2 x cp2 | 4 | 12.57299 | 7.50029 | 3.35914 |
| dp2 x cp4 | 8 | 12.53546 | 7.32970 | 3.38800 |

Two boundaries raise instead of running: `Q_LEN` not divisible by `cp * 128`, and a folded microbatch wider than the context window. Not in this PR: CP inside the vision tower and the report's dynamic CP for large images. Without `context_parallel_degree > 1` none of this executes.

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py            +265  the two CP contracts, the head/sequence
                                   all-to-all, MLA's Ulysses body and the
                                   packed-document mask rebuild it needs
      kda.py                 +119  KCP on the KDA layers: conv halo exchange and
                                   the prefix scan over the recurrent state, via
                                   attention-gym's fla CP wrappers
      model.py             +115/-15 MLA branches to Ulysses when a CP group is set;
                                   overrides _validate_cp_backend; the vision
                                   splice follows the sequence shard under CP
      parallelize.py         +68/-5 apply_cp_kimi_k3: wires the group onto both
                                   layer kinds, checks head divisibility, and fails
                                   at wiring time if attention-gym's fla CP
                                   wrappers are missing
    torchtitan/models/common/decoder.py  +7/-3  the spmd_types requirement becomes
                                   an overridable method
    torchtitan/distributed/fsdp.py       +24   add_zero_valued_dependency: keeps a
                                   partly consumed FSDP module in the graph
    tests/
      unit_tests/cpu/test_kimi_k3_cp_contracts.py     +63  the folded-layout contracts
      unit_tests/cpu/test_kimi_k3_cp_document_mask.py +89  two-rank gloo: the gathered
                                   mask refuses both boundary attentions
      integration_tests/features.py                    +8  the cp2 cell
    torchtitan_recipes/tests/features.py              +13  its configuration

### CI/CD Coverage

CPU contract tests for the two things that used to fail silently, a two-rank document-mask test, and a cp2 integration cell.

### Numerical Correction run with unmerged upstream grad-norm precision forced to FP32

The same matrix with the grad-norm reduction carried in float32 (https://github.com/pytorch/torchtitan/pull/4135, a separate change not on this branch): dp2 x cp2 is bitwise the bf16-grad-norm run, cp2 and cp4 move at step 3 (7.04577 to 7.10562, 6.89080 to 7.29175), and the remaining cells move only at step 10 -- the same grouping-sensitivity shape the pre-fix companion showed.

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.60544 | 7.30226 | 3.23705 |
| cp2 | 2 | 12.53996 | 7.10562 | 3.38411 |
| cp4 | 4 | 12.53406 | 7.29175 | 3.36449 |
| cp8 | 8 | 12.53711 | 7.52113 | 3.49337 |
| dp2 | 2 | 12.58193 | 7.44923 | 3.32140 |
| dp2 x cp2 | 4 | 12.57299 | 7.50029 | 3.35914 |
| dp2 x cp4 | 8 | 12.53546 | 7.32970 | 3.39451 |

### Review round: the KCP machinery is imported from attention-gym

Following the dispatch-to-FLA direction from review, the two CP-specific imports (`build_cp_context`, `causal_conv1d_cp`) now come from `attn_gym.linear.kda.fla_cp` -- the attention-gym port of fla's CP machinery (prefix-scan context, conv halo, fragment entry; https://github.com/meta-pytorch/attention-gym/pull/421) -- instead of reaching into fla-core's module layout directly; fla remains the underlying kernel provider, and the wiring-time availability check names the attention-gym module.

The full matrix re-run on this dependency reproduces six of the seven cells bitwise at steps 1/3/10 (dp1, cp2, cp8, dp2, dp2 x cp2, dp2 x cp4), confirming the wrappers are pass-throughs. The seventh, cp4, is launch-nondeterministic independently of this change: its step-1 forward flips between exactly the two values the two tables above already print (12.52432 and 12.53406 -- a difference grad-norm precision cannot produce), and one re-run invocation reproduced both, one per pass, on the same seed and checkpoint.

### Review round: packed-document boundaries

Review ask: how are document boundaries preserved under CP? The MLA path was rebuilding a causal-only mask for the sequence Ulysses reassembles and used the context window to reject streams it could not mask; that guard never fired here because this dataset packs two documents into exactly one context window. Fixed: after the all-to-all every rank holds the full sequence, so the CP path now gathers the contiguous positions shards and builds the same causal x document mask the non-CP path uses; the guard, its config plumbing and the shape-keyed mask cache are gone. A two-rank gloo test packs three documents with one boundary on the shard cut and one inside a shard: the gathered mask equals the mask built from the global positions, and both boundary attentions are refused where a causal-only mask lets them through. The tables above are re-measured on the fixed head. KCP keeps `cu_seqlens = [0, total]` deliberately: the non-CP KDA path passes no boundaries either (KDA sample packing lands with PR-4347), so the CP path matches the single-rank recurrence semantics rather than diverging from them.
