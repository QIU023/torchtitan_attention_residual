# PR title: kda: an fla backend, with context parallelism

Target: meta-pytorch/attention-gym. Branch `kda-fla-backend` (`819370f`, two commits) on QIU023/attention-gym. Evidence: `phase13_k3like_48b_posttrain/ATTN_GYM_FLA_BACKEND_2026-08-29.md`. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Adds `Impl.FLA`: `chunk_kda` over flash-linear-attention's triton kernels, sitting between the SM100-only fused path and the pure-PyTorch reference -- arch-agnostic like reference, kernel-grade like fused. This picks up the "dispatch to FLA on unsupported archs" direction from the kda CP example discussion; `fla` is an optional dependency (importing the backend without it raises with an install pointer), the same standing it has in torchtitan's qwen3_5 today. A second commit exposes fla's own context-parallel machinery behind this package's conventions, so KDA runs context-parallel on any CUDA architecture.

### Design

- `Impl` gains `FLA`; the `chunk_kda` dispatch gains one branch; the adapter lives in `kda/fla_backend.py`.
  - The adapter feeds fla the exact contract this package exposes -- q/k pre-normalized, `gate` the bound natural-log decay, `beta` already sigmoided -- so every `use_*_in_kernel` switch stays off. No convention translation is needed except one: the two libraries transpose the recurrent state's last two axes relative to each other, and the adapter owns that flip in both directions (`initial_state` in, `final_state` out).
- `kda/fla_cp.py` wraps fla's CP mechanism: `build_fla_cp_context` (the delta recurrence becomes a prefix scan over rank-local fragments; `cu_seqlens` carries global packed-document boundaries, defaulting to one document), `causal_conv1d_cp` (the short-convolution halo exchange), and `chunk_kda_cp` (this rank's fragment through the scan).
  - `initial_state`/`output_final_state` are unsupported under CP: the final state only matters for decoding, which does not run context-parallel.
- `bound_gate` is unchanged: the fla backend consumes its reference-impl output.

### Results

FLA against REFERENCE on the same bf16 inputs (B=2, T=128, H=4, K=V=128, RTX 5060 Ti / SM120 -- no Blackwell-DC hardware involved):

| check | max diff |
|---|---|
| forward output | 2.4e-4 |
| final state (after the axis flip; 0.61 before it) | 2.9e-3 |
| FLA half-run resumed from its own final state vs one full run | 0.0 (bitwise) |
| REFERENCE final state resumed by FLA (cross-impl handoff) | 1.2e-4 |
| gradients dq/dk/dv/dgate/dbeta | <= 7.5e-9 |
| varlen, two documents via `cu_seqlens` | 2.4e-4 |

Context parallel, two ranks over contiguous shards of T=256 against the single-rank run:

| check | max diff |
|---|---|
| forward, each rank's fragment vs the full run's slice | 6.1e-5 |
| backward through the cross-rank scan | gradients finite, norms match |

The cross-impl state handoff means CP orchestration built on state composition can run its per-rank compute on FLA without the composition layer noticing.

### Changed files

    attn_gym/linear/
      types.py           +1     Impl.FLA
      kda/api.py         +14    the dispatch branch
      kda/fla_backend.py +81    the adapter and the state-axis flip (new)
      kda/fla_cp.py      +105   CP: prefix-scan context, conv halo, fragment entry (new)

--- PASTE END ---
