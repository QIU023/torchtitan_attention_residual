# The declarations PR under both backends, same data (2026-09-06)

tianyu-l on 4492 (2026-09-05): "numerics difference look too huge, are you testing with identical
global batch size across?" The table he read had dp1 / dp2 / dp2 x ep2 rows under `spmd_types`
and only dp1 under `partial_dtensor`. Every cell trains on 8192 tokens per step, but the rows do
not read the same samples: the loader shards documents by dp rank
(`torchtitan/components/data/sources.py:187`, `split_dataset_by_node`; `dataset.py:156`,
`shard_start = dp_rank * shard_size`), so dp1 and dp2 are different data streams from step 1 on
(step-1 loss 12.52977 against 12.53137 with identical weights). The comparison the PR needs is the
same cell under both backends.

## Step-1 gradients, every parameter (`decl_grad_dump.sh`, `cmp_backend_grads.py`)

Tree `/tmp/wt_decldump` = `dbc60701d` + the run worktree's local hacks (SM120 guard lift, registry
alias) + a dump-and-exit hack (rank 0 saves every parameter's step-1 gradient in its own dtype before
clipping). Seed checkpoint and batch of the declarations matrix. Each cell twice on one inductor
cache (`warm`, then `dump`).

| pair | parameters bitwise | sign flips | note |
| --- | --- | --- | --- |
| dp2: partial_dtensor vs spmd_types | 750 / 750 | 0 of 1.12e9 | cold cache and warm cache alike |
| dp2 x ep2: partial_dtensor vs spmd_types | 750 / 750 | 0 of 1.12e9 | same |
| dp2_pd: cold cache vs warm cache | 750 / 750 | 0 | the compile cache does not touch step 1 |
| dp2_ep2_pd: cold vs warm | 750 / 750 | 0 | same |
| dp2 vs dp2 x ep2 (partial_dtensor, same samples) | 0 / 750 | 1.371% | the reorder floor: median relative norm difference 5.8e-2; experts 1.71% flips, KDA 2.54%, MLA 1.22%, router 1.75%, vision 1.54%, the rest 0.29% |

So the two backends are the same computation at step 1, on both cells. Turning EP on with the
same samples moves 1.4% of the gradient elements' sign at step 1: a different grouping of the
expert GEMMs rounds differently in bf16 and the random-init router flips on it (the mechanism of
`PP_STEP10_SPREAD_2026-09-04.md`), which is where the dp2 / dp2 x ep2 rows' later gap comes from.

## Loss rows on one compile cache (`rebase_main_decl_pairs.sh`)

Same tree as the declarations matrix (`/tmp/wt_declrun`), seed and batch; the six cells in one
mx3 invocation so they share the inductor cache. dp1 under `spmd_types` came out
12.52977 / 7.27107 / 2.98077, the numbers of the 09-04 matrix on `96eeb51c1`; the 09-05 matrix on
the same tree (`dbc60701d`, the logbook body's rows) had 12.52977 / 7.36833 / 2.91045. Same seed,
same samples, step-1 gradients bitwise across caches (above): the compile cache picks other
autotuned kernels from step 2 on and this flavor (bf16 end to end, lr 8e-4, 2-step warm-up)
amplifies the rounding into 1.3% at step 3 and 2.4% at step 10. That row is the noise floor the
table shows.

| cell | partial_dtensor | spmd_types |
| --- | --- | --- |
| dp1 | 12.52977 / 7.27107 / 2.98077 | 12.52977 / 7.27107 / 2.98077 |
| dp2 | 12.53137 / 7.31248 / 3.15823 | 12.53137 / 7.31248 / 3.15823 |
| dp2 x ep2 | 12.53146 / 7.20212 / 3.10296 | 12.53146 / 7.20212 / 3.10296 |

Every pair bitwise through step 10, not only at step 1: on one compile cache the two backends are
the same computation for the whole run. The 09-04 matrix's rows (the GitHub body) are these numbers;
the 09-05 matrix's rows (the logbook body until today) were the other cache's. Run dir
`/workspace/mx3_decl_pairs_0906_005832`, dumps under `/workspace/decl_dump`.
