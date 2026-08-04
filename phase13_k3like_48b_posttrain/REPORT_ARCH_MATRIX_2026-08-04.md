# Parallelism matrix on the report's architecture

Flavor `kimi_k3_debugmodel_report_arch`: the eager reference's debug config with
one entry changed -- layer 13 is Gated MLA, so the final layer performs global
attention. Report sec 2.1 places that extra layer at the end of the backbone,
and the released `config.json` lists `full_attn_layers` ending `[..., 92, 93]`.

This is the primary matrix. `TWIN_MATRIX_2026-08-04.md` carries the same cells
on the eager reference's architecture unchanged, plus the defect history and
the retractions.

Multimodal (MoonViT-V2 + backbone), seed 42, `--debug.deterministic`, global
batch 8, `local_batch_size` 1 so gradient accumulation is on, 10 steps.

## Eager

| configuration | step 1 | step 5 | step 10 |
|---|---|---|---|
| `single-GPU` | 12.05342 | 11.00553 | 9.90565 |
| `fsdp2` | 12.05033 | 11.06650 | 9.91598 |
| `pp2` | 12.04891 | 10.79037 | 9.80817 |
| `cp2` | 12.04192 | 11.05030 | 9.91244 |
| `tp2` | 12.07846 | 11.02278 | 9.90544 |
| `fsdp2 x tp2 x pp2` | 12.05989 | 10.64503 | 9.73071 |
| `fsdp2 x tp2 x cp2` | 12.10048 | 11.06030 | 9.93935 |
| `tp2 x pp2 x cp2` | 12.05395 | 10.61264 | 9.72969 |
| `fsdp2 x pp2 x cp2` | 12.03465 | 10.61977 | 9.72763 |
| `ep2 x fsdp2` | 12.05033 | 11.02921 | 9.91933 |
| `ep2 x fsdp2 x tp2 x pp2` | 12.05717 | 10.75712 | 9.76465 |
| `ep2 x fsdp2 x tp2 x cp2` | 12.06437 | 11.09464 | 9.95156 |
| `ep2 x fsdp2 x pp2 x cp2` | 12.07291 | 10.68421 | 9.76349 |
| `ep8 x fsdp8` | 12.01828 | 11.03459 | 9.92187 |
| `pp4` | 12.08035 | 10.79280 | 9.80668 |
| `pp8` | 12.04015 | 10.81690 | 9.80949 |
| `tp4` | 12.04172 | 11.10711 | 9.95002 |
| `cp4` | 12.03731 | 11.00092 | 9.91970 |

**18/18.** Step-1 spread 12.0183 to 12.1005, against `ln(163840) = 12.006`.

`tp4` and `cp4` reproduce their 3-step values exactly at step 1, which is the
cheap consistency check on the re-run. The thirteen degree-2 cells also ran 100
steps clean on this flavor; those numbers are in `TWIN_MATRIX_2026-08-04.md`
pending a re-run of all eighteen at that horizon.

## Compiled

15/18 as-is, 18/18 with the shim.

The three gaps are one upstream operator limitation, not ours:
`torch._grouped_mm` rejects an operand whose contraction dimension is zero,
which is the weight-gradient shape when a rank holds no routed tokens at all.
Five-line reproduction in `matrix_scripts/grouped_mm_repro.py` -- no model, no
compile, no distributed. With `matrix_scripts/gmm_shim.py`, which re-strides
the empty operand exactly as the patch proposed upstream would, all eighteen
pass. **The shim is ours and upstream does not have it, so 15/18 is the number
reported.**

Worth recording because it corrects an earlier inference: `ep8_fsdp8` passes
compiled WITHOUT the shim. One expert per rank guarantees empty expert groups
every step, and that is not sufficient to trigger the defect -- the failing
shape needs a rank with zero routed tokens in total, which the ep2 combinations
with TP/PP/CP produce and ep8 does not.

## Not expressible on this config

Recorded rather than worked around by editing the flavor, since the point of
running the eager reference's config is that it is the eager reference's
config:

* `tp8`, `cp8`, `tp4 x cp4` -- 4 attention heads (MLA and KDA both) to shard.
* `pp8 x vp4` -- 13 layers cannot host 32 virtual stages.

High-degree coverage on our own flavors (tp8, cp8, PP8xVP4, EP@896) exists
separately in this logbook.

## Environment

    torch      2.14.0.dev20260802+cu130 (CUDA 13.0)
    fla-core   0.5.1
    GPU        8 x RTX 5060 Ti, 15.5 GiB, sm_120, 101376 B opt-in shared memory

Which cells can run is hardware-dependent; the losses are not. See
`matrix_scripts/README.md` for the three findings that are specific to this
card and would not reproduce on A100 or H100.
