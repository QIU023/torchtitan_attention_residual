# Migration step 3, first axis: context parallel on the upstream K3 model

Branch `migrate_step3_cp` at `df09ce78b`, five commits on `align_4025`.

Upstream's `parallelize.py` is 100 lines and rejects CP outright. This adds it for the
hybrid stack: Ulysses for MLA, KCP for KDA, both attached by replacing `forward` so the
vendored model file stays byte-identical and rebases stay cheap.

## Results

| gate | result |
|---|---|
| all-MLA flavor: `fsdp2` control | 10/10, byte-identical across three tree revisions |
| all-MLA: `cp2` | 10/10 |
| all-MLA: `fsdp2_cp2` | 10/10 |
| hybrid (15 KDA + 6 MLA): `fsdp2` control | `8.05379 -> 3.00021` |
| **hybrid: `cp2`, Ulysses and KCP together** | **`8.04780 -> 3.01302`, 10/10** |
| guard: KDA under CP before KCP existed | rejected, for the KDA reason |
| guard: `local_batch_size != 1` under KCP | rejected, for the batch-size reason |

| correctness, module vs single rank | result |
|---|---|
| Ulysses MLA, fp32 | **3e-7 relative** |
| KCP KDA, bf16 | **bf16 noise**, both ranks at the same absolute difference |

## Why not core's ring path

Core's `apply_cp_to_forward` routes an SDPA inner attention onto the CP dispatcher. K3's
MLA lands there and fails inside the dispatcher's accumulation with `aten.add.Tensor got
mixed torch.Tensor and DTensor`. The control separates the cause: **llama3 passes `cp2` on
the same tree and the same box**, so ring works here and this attention is what it does not
handle. The KIND probe shows q, k and v all arriving as `DTensor(S(1))`, so the mixing is
inside the dispatcher rather than in the wiring.

Ulysses is a different decomposition, not a fallback:

| | Ulysses | ring |
|---|---|---|
| what the kernel sees | an ordinary unsharded problem | sharded, needs accumulation support |
| K3 MLA | works | fails |
| bound | `cp <= num_heads` -- real for K3 | none |
| composes with a sequence-sharded pipeline | no | yes (this is KCP's axis) |

Both are kept. **Upstream torchtitan implements Ulysses nowhere** -- a grep over
`torchtitan/` outside our own model folder returns nothing -- so it is our increment, not a
duplicate of something core provides.

## Reuse, not rewrite

`build_kcp_context` and `conv_with_halo` are imported from `kimi_k3.kcp`, the imperative
implementation. `conv_with_halo` grew one optional `activation` argument, because fla's
`ShortConvolution` carries `conv.activation` and the upstream model uses a plain `Conv1d`
with SiLU applied outside. When `kimi_k3` is retired those helpers move; they do not get
rewritten.

The fla entry points were checked rather than assumed: fla 0.5.1 has `build_cp_context`,
and `chunk_kda` does accept `cp_context`.

## Constraints found, both now guarded

* **KCP requires `local_batch_size == 1`.** fla's `causal_conv1d_cp` asserts `[1, T, D]` --
  the CP path is built around cu_seqlens packing, where the batch is one packed sequence.
  Checked at parallelize time rather than several minutes later at the first KDA forward.
* **`chunk_kda` takes `A_log` and `dt_bias` through `**kwargs`**, so a misspelled keyword
  is dropped silently instead of raising.

## Two judges that had to be fixed before they judged anything

Both failures were the measurement, not the thing measured. Recording them because the
shape recurs.

**End-to-end loss parity is not a CP correctness check.** Train `dp1`, train `cp2`, compare
losses -- under it the ported Ulysses showed a 5.1e-4 step-1 gap, which looked like a
failure. The control killed it: **our own Ulysses, the one whose docstring claims
bit-exactness against a single-rank reference, shows a LARGER gap (8.6e-3)** under the same
harness. `dp1` and `cp2` are not the same computation at the trainer level -- CP reorders
the sequence for head-tail load balancing and the batch composition differs. Without that
control I would have spent the next hour fixing something that was not broken.

**A relative error against a diverged reference is not a measurement.** The first KCP probe
printed `PARITY PASS` while the reference output was ~1e25: it hand-rolled an
initialization that skipped the 1-D parameters, leaving `A_log` and `dt_bias` on
uninitialized memory, and 6.5e-3 relative against an exploding scale sat under the
threshold. Both probes now assert the reference is finite and of sane magnitude before
computing anything.

Also found: the probe those `kcp.py` docstrings cite, `kda_ulysses_cp_probe`, **exists
nowhere in the tree**, so the bit-exactness claim could not be re-run. `ulysses_mla_parity.py`
and `kcp_kda_parity.py` close that gap.

## What made the KCP result trustworthy beyond its number

Rank 0 has no left neighbour, so its convolution halo is trivially correct, and a broken
halo or prefix scan would make rank 1 strictly worse. Both ranks report the same absolute
difference, 1.221e-04. The structure of the result rules out the failure modes; the
magnitude alone would not have.

## Ahead on this step

LoRA / MXFP4 next. The attachment points are mapped and one difference is worth carrying
forward: **our KDA reads `linear.weight` directly for the fla kernels, so LoRA wrappers
there are silently dead** -- which is why `apply_lora` skips the KDA subtree structurally.
Upstream's KDA goes through the module (`self.q_proj(x_BLD)`), and so does the KCP forward
written here, so on this tree KDA projections are targetable and that skip does not port.

Three Linears must be excluded: `attention_res_proj`, `ffn_res_proj`, `output_res_proj`.
They are the zero-initialized AttnRes pseudo-queries, and the carrier story depends on
small deltas accumulating from exactly zero.

Then DEP, then PP with the adapter last.

---

# PP: the carrier migration, stopped at a verified intermediate

Branch `migrate_carrier_tensor` at `77adb091b`, one commit on `align_4025`.

## Done and verified

`block_attn_res_tensor` and `KimiAttnResDecoderLayer.forward_tensor_carrier` take the block
history as one `[T, N, D]` tensor instead of a list plus a separate partial, with the
running partial sum riding inside the hidden state. Three Python accumulators become two
tensors, both in the signature.

The gate was **bitwise equality**, not the matrix: a pure container change has no licence to
move a digit, and the 54-cell matrix could not distinguish a container bug from the drift it
tolerates elsewhere. Measured on one GPU on `report_arch` -- blocks committed 2 both ways,
carrier and hidden state both bitwise equal (`matrix_scripts/carrier_equivalence_probe.py`).

Both paths live side by side. The gated graft keeps the list path with an explicit
`NotImplementedError`, since its `plain_stream` is a third accumulator this form does not
carry; it is off in all three matrix arms, so it does not block the 54.

## Why the matrix was NOT run on this

The model's forward loop still calls the list path, so a 54-cell run would re-run the old
code and report green without touching the change. A passing matrix that does not exercise
the diff is worse than no matrix, and it is the thing to refuse rather than the thing to
report.

## The blocker, precisely

Switching the model loop means switching the PP wire format at the same time, because the
model's return value IS the P2P payload. And there:

**`[T, N, D]` flattens `B * L`, while the adapter slices blocks as `[B, T, D]`.**
`unstack_blocks` cannot invert the flatten without being told B and L. This is information
loss, not a reshape.

Three ways out, and the choice determines the semantics of three places in the adapter's
1571 lines (`RankLocalCache`, `_LocalCacheCapture`, `_install_augment_hook`):

1. carry `B`/`L` as metadata alongside the tensor -- smallest change, adds a second thing
   that must cross the wire in step;
2. move the adapter to column-wise operation on `[T, N, D]` -- largest change, ends at the
   upstream shape exactly;
3. keep a 4-D wire format `[N, B, L, D]` and use the 3-D form only inside the block --
   no adapter change, but the carrier crossing the wire is then not the upstream shape,
   so the declarative win stops at the stage boundary.

(2) is the only one that reaches the end state the migration is for. (3) is the one that
gets a green 54 soonest and would have to be redone.

## Next concrete action

Pick between those three, then: switch the model loop, switch the wire format, run the
three arms, and report `54/54` with the per-cell table. The PP cells are the risk surface --
`pp2`, `fsdp2_tp2_pp2`, `tp2_pp2_cp2`, `fsdp2_pp2_cp2`, `ep2_fsdp2_tp2_pp2`,
`ep2_fsdp2_pp2_cp2` plus maxdeg `pp4`/`pp8` -- and if they break it is the adapter's column
semantics, not CP or EP.
