# FSDP x PP x CP x EP: numerical baseline from a shared seed (2026-07-30)

First time this repo can claim cross-parallelism NUMERICAL agreement rather than
just "it runs, rank-identically". Every leg loads the same step-0 seed checkpoint,
so the arms start from identical weights -- without that, FSDP2's meta-init gives
each rank its own RNG stream and two parallel degrees are simply different models.

TP is deliberately excluded: it carries an unattributed one-directional gradient
gap (TP_GRAD_FINDING_2026-07-29), and mixing a known open question into this
matrix would make the other axes unreadable.

k3mini, 3 steps, global batch 8, seq 512, seed 42 deterministic.

## Results: 12 of 13 legs pass

| leg | GPUs | step-1 loss | grad_norm |
| --- | --- | --- | --- |
| `fsdp2` | 2 | **7.71304** | 8.4957 |
| `fsdp2_pp2` | 4 | **7.71304** | 8.4957 |
| `ep2_fsdp2` | 2 | **7.71304** | 8.4957 |
| `ep2_fsdp2_pp2` | 4 | **7.71304** | 8.4957 |
| `cp2` | 2 | **7.71793** | 8.7290 |
| `pp2_cp2` | 4 | **7.71793** | 8.7290 |
| `fsdp2_cp2` | 4 | **7.71135** | 8.5883 |
| `fsdp2_pp2_cp2` | 8 | **7.71135** | 8.5883 |
| `ep2_fsdp2_cp2` | 4 | **7.71135** | 8.5883 |
| `ep2_fsdp2_pp2_cp2` | 8 | **7.71135** | 8.5883 |
| `ep2_fsdp4` | 4 | 7.71541 | 8.7562 |
| `pp2` | 2 | FAIL (hardware, below) | -- |

## What the grouping says

The legs fall into three groups, and every group is bit-identical internally.

**PP is numerically transparent.** `fsdp2` = `fsdp2_pp2`, `cp2` = `pp2_cp2`,
`fsdp2_cp2` = `fsdp2_pp2_cp2` = `ep2_fsdp2_pp2_cp2`. Adding pipeline parallelism
does not move a digit. That is the strongest correctness evidence yet for the PP
cross-stage adapter, which is this repo's core asset.

**EP is numerically transparent.** `ep2_fsdp2` = `fsdp2`,
`ep2_fsdp2_cp2` = `fsdp2_cp2`, `ep2_fsdp2_pp2` = `fsdp2_pp2`. Turning expert
parallelism on or off does not move a digit either.

**CP does change the numbers** (7.71304 -> 7.71135 with FSDP, -> 7.71793 without),
which is expected and not a defect: CP changes the reduction order and, because EP
and CP are both carved out of the data-parallel axes, it changes the dp grouping
too. `ep2_fsdp4` sits alone for the same reason -- its dp degree is 4, not 2.

## The one real failure is the GPU, not the code

`pp2` alone (dp_shard=1, pp=2):

    tvm.error.InternalError: Failed to set the allowed dynamic shared memory
    size to 108160

The RTX 5060 Ti offers 101,376 B (`shared_memory_per_block_optin`). Same ceiling
already documented for `kda_head_dim=64`; H100/H200 have 227 KB and are
unaffected. It reproduces at local-batch 4 as well.

Note `pp2_cp2` PASSES while `pp2` alone fails. That is consistent rather than
contradictory: CP halves the local sequence, which changes the KDA kernel's shape,
so fla's autotuner selects a configuration that fits. It confirms the failure is
shape-dependent kernel selection against a hardware ceiling, not a logic error on
the PP-only path.

## Correction to the matrix run's own report

The driver first reported `ep2_fsdp2_pp2_cp2` as FAIL. Re-running it in isolation
gives `step: 1 loss: 7.71135 grad_norm: 8.5883` with ZERO tracebacks -- exactly
the group it belongs to. The FAIL was the driver's rank-identity heuristic
interacting with PP's negative placeholder losses, i.e. a reporting artifact. The
four-way combination works.

Two harness lessons, both already paid for once in this logbook: a PASS/FAIL
heuristic over log lines needs its own control, and a leg reported as failing must
be reproduced in isolation before it is believed.
