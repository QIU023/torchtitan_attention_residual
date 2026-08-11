"""Is KDA's chunked kernel exact under the two things CP does to it?

The CP/TP gradient question has been open because every instrument available compared a
sharded arm against a reference whose SHAPE differed, and `chunk_kda` is a chunked scan
whose blocking follows the input shape. So a difference could never be attributed.

`fla.ops.kda.naive.naive_recurrent_kda` breaks that: it is token-by-token with no chunk
size, so it is shape-independent by construction, and it is fla's own reference rather than
a recurrence rederived here.

Two questions, in the order they have to be asked:

1. **Is the reference usable?** It has to agree with `chunk_kda` on one rank first.
   Anything else makes it another untrusted instrument, which this project has already
   paid for three times.
2. **Does splitting the sequence change the answer?** CP shards the sequence and carries
   the recurrent state across the boundary. `chunk_kda` expresses that directly through
   `initial_state` / `output_final_state`, so it can be tested on ONE GPU with no process
   group at all -- which removes every confound a distributed comparison brings.

Inputs must be realistic or the test is vacuous. The first version used `g` near zero
(no decay) and unnormalised q/k; the state then explodes to ~1e17 and both
implementations return garbage that agrees to 1e-2. KDA l2-normalises q/k and its gate is
`-exp(A_log) * softplus(...)`, strongly negative, so that is what this drives.

Usage: python kda_shape_independent_probe.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda
from fla.ops.kda.naive import naive_recurrent_kda


def _inputs(batch: int, seq: int, heads: int, key_dim: int, value_dim: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    return (
        F.normalize(torch.randn(batch, seq, heads, key_dim, device=device), dim=-1),
        F.normalize(torch.randn(batch, seq, heads, key_dim, device=device), dim=-1),
        torch.randn(batch, seq, heads, value_dim, device=device),
        -F.softplus(torch.randn(batch, seq, heads, key_dim, device=device)) - 0.5,
        torch.rand(batch, seq, heads, device=device).sigmoid(),
    )


def _out(result):
    return result[0] if isinstance(result, tuple) else result


def main() -> None:
    batch, heads, key_dim, value_dim = 1, 2, 32, 32

    print("1. is the reference usable? chunk_kda vs naive_recurrent_kda, one rank, fp32")
    for seq in (16, 64, 128, 256):
        q, k, v, g, beta = _inputs(batch, seq, heads, key_dim, value_dim, seed=0)
        ref = _out(naive_recurrent_kda(q, k, v, g, beta))
        got = _out(chunk_kda(q, k, v, g, beta))
        scale = max(ref.abs().max().item(), 1e-12)
        print(
            f"   seq={seq:4}  ref_max={scale:8.5f}  rel={(got - ref).abs().max().item() / scale:.3e}"
        )

    print()
    print("2. does sharding the sequence change the answer? (what CP does to the scan)")
    seq = 256
    q, k, v, g, beta = _inputs(batch, seq, heads, key_dim, value_dim, seed=0)
    ref = _out(naive_recurrent_kda(q, k, v, g, beta))
    full = _out(chunk_kda(q, k, v, g, beta))

    half = seq // 2
    first, state = chunk_kda(
        q[:, :half],
        k[:, :half],
        v[:, :half],
        g[:, :half],
        beta[:, :half],
        output_final_state=True,
    )
    second, _ = chunk_kda(
        q[:, half:],
        k[:, half:],
        v[:, half:],
        g[:, half:],
        beta[:, half:],
        initial_state=state,
        output_final_state=True,
    )
    split = torch.cat([first, second], dim=1)

    scale = max(ref.abs().max().item(), 1e-12)
    print(f"   reference magnitude        {scale:.5f}")
    print(f"   full        vs reference   rel {(full - ref).abs().max().item() / scale:.3e}")
    print(f"   two shards  vs reference   rel {(split - ref).abs().max().item() / scale:.3e}")
    print(f"   full        vs two shards  rel {(full - split).abs().max().item() / scale:.3e}")


if __name__ == "__main__":
    main()
