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
