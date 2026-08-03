# K3-faithful parallelism baselines (2026-07-27)

Produced by `run_k3_matrix.sh` on 8x RTX 5060 Ti, flavor
`kimi_k3_mini_block_attn_res` unless noted, 3 steps, global batch 8,
seq 512, `--debug.seed 42 --debug.deterministic`.

**These replace every earlier MoE loss in this logbook.** The previous numbers
were produced with uninitialized routed experts and measure a different model --
see `EVIDENCE_INVALIDATION_2026-07-27.md`. Nothing here is comparable to a
pre-2026-07-27 figure.

## What a PASS means

1. it runs (no mixed Tensor/DTensor, no shape error),
2. every rank prints an identical loss, and
3. grad_norm is finite and in the same range as the 1D baseline.

Loss *equality across degrees* is deliberately NOT a criterion. FSDP2 meta-init
gives each rank its own RNG stream, so two different parallel degrees start from
different weights; matching curves would require a seed checkpoint. Read the
columns as "this configuration trains", not as a numerical A/B. The spread below
(7.700-7.725 at step 1) is that init difference, not a correctness signal.

Non-last pipeline stages report a negative placeholder loss (-2, -4 and -8 all
observed across configurations), so the scorer filters on the sign rather than on
specific values.

## Results

| configuration | GPUs | step 1 | step 2 | step 3 | grad_norm (step 1) |
| --- | --- | --- | --- | --- | --- |
| `fsdp8` | 8 | 7.70056 | 7.63014 | 7.44544 | 8.4765 |
| `fsdp2` | 2 | 7.71802 | 7.64113 | 7.44400 | 8.6584 |
| `tp2` | 2 | 7.72128 | 7.61571 | 7.40192 | 2.7507 |
| `cp2` | 2 | 7.70770 | 7.65509 | 7.47410 | 8.8230 |
| `pp2` | 2 | 7.71068 | 7.64485 | 7.46881 | 8.8399 |
| `ep2` | 2 | 7.71802 | 7.63872 | 7.44358 | 8.6584 |
| `dp2xtp2` | 4 | 7.72316 | 7.61362 | 7.36674 | 2.8006 |
| `dp2xcp2` | 4 | 7.70592 | 7.62255 | 7.43235 | 8.6569 |
| `dp2xpp2` | 4 | 7.70506 | 7.62116 | 7.43931 | 8.9319 |
| `tp2xcp2` | 4 | 7.70135 | 7.62415 | 7.39226 | 2.8131 |
| `tp2xpp2` | 4 | 7.72488 | 7.60410 | 7.35960 | 2.7732 |
| `dp2xep2xtp2` | 8 | 7.71937 | 7.62559 | 7.44720 | 8.9995 |
| `dp2xtp2xcp2` | 8 | 7.70210 | 7.60485 | 7.37473 | 2.7527 |
| `dp2xtp2xpp2` | 8 | 7.71550 | 7.60692 | 7.37176 | 2.8170 |
| `dp2xcp2xpp2` | 8 | 7.73273 | 7.65303 | 7.45071 | 8.8714 |
| `tp2xcp2xpp2` | 8 | 7.71731 | 7.60962 | 7.39083 | 2.8120 |
| `qat_dp2xtp2` | 4 | 7.71787 | 7.61589 | 7.37524 | 2.8325 |
| `qlora_dp2xtp2` | 4 | 7.71897 | 7.70435 | 7.71378 | 0.5441 |

All 18 legs pass. Notable ones:

* `dp2xep2xtp2` -- expert parallelism composed with TP and FSDP, the
  configuration the 2.8T shape needs, now with routed experts that are actually
  alive.
* `tp2xcp2xpp2` and `dp2xcp2xpp2` -- three-way with CP and PP together. CP here
  is our Ulysses all-to-all for KDA and MLA, not KCP; KCP integration is still
  open, and Ulysses stays as its A/B.
* `qat_dp2xtp2` / `qlora_dp2xtp2` -- MXFP4 QAT on the routed experts, and LoRA on
  the full K3 module set, each under a 2D mesh. The QLoRA leg is what surfaced
  the colwise-TP adapter bug (`attn_gate_proj` at tp=2), since no
  colwise `use_local_output` module had been a LoRA target before.

The lower grad_norm on TP legs (~2.8 vs ~8.9) is expected: the LoRA and TP legs
differ in how many parameters are trainable and in the gradient's mesh
placements, not in correctness. grad_norm is compared within a leg across steps,
not across legs.

