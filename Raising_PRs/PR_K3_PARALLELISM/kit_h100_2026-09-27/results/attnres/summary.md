## identity (debug model, 2048 tokens/step in 512-token micro-batches, 10 steps, one warm cache)

| tree | AC | rc | step 1 loss / gn | step 10 loss / gn | steps equal to main | peak memory (max reserved) | step 10 mem | tps steps 6-10 |
|---|---|---|---|---|---:|---:|---:|---:|
| main | none | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.66 GiB | 0.66 | 1522 |
| l4656 | none | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.66 GiB | 0.66 | 1556 |
| c4780 | none | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.56 GiB | 0.56 | 1561 |
| main | selective | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.35 GiB | 0.35 | 1054 |
| l4656 | selective | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.35 GiB | 0.35 | 1002 |
| c4780 | selective | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.35 GiB | 0.35 | 1001 |
| main | full | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.31 GiB | 0.31 | 1032 |
| l4656 | full | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.30 GiB | 0.30 | 1036 |
| c4780 | full | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 0.30 GiB | 0.30 | 1127 |
| main, fresh cache | none | rc=0 | `8.06529` / `2.3906` | `6.09957` / `8.3125` | 10 / 10 | 1.01 GiB | | 2086 |

## AC off, debug model, one micro-batch (3 steps; peak = max reserved over the steps)

| config | main peak | l4656 peak | c4780 peak | saved (c4780 vs main) | step 3 loss / gn equal | tps main / l4656 / c4780 |
|---|---:|---:|---:|---:|---|---:|
| 2048 | 1.88 GiB | 1.86 GiB | 1.41 GiB | 0.47 GiB (25%) | yes | 5860 / 6731 / 4979 |
| 4096 | 3.53 GiB | 3.52 GiB | 2.66 GiB | 0.87 GiB (25%) | yes | 12385 / 13149 / 12111 |
| 8192 | 6.75 GiB | 6.73 GiB | 4.98 GiB | 1.77 GiB (26%) | yes | 21449 / 23715 / 23186 |
| 16384 | 13.26 GiB | 13.21 GiB | 9.71 GiB | 3.55 GiB (27%) | yes | 47688 / 52175 / 49718 |

## list carrier: main vs #4656, 48 layers at dim 2048

| config | main peak | l4656 peak | saved (l4656 vs main) | step 3 loss / gn equal | tps main / l4656 |
|---|---:|---:|---:|---|---:|
| selective_b4_8192 | 30.95 GiB | 29.58 GiB | 1.37 GiB (4%) | yes | 4436 / 4554 |
| selective_b4_16384 | 55.30 GiB | 52.54 GiB | 2.76 GiB (5%) | yes | 5411 / 5444 |
| selective_b12_8192 | missing | missing |  |  | - / - |
| selective_b12_16384 | missing | missing |  |  | - / - |
| full_b4_8192 | missing | missing |  |  | - / - |
| full_b4_16384 | missing | missing |  |  | - / - |
| full_b12_8192 | missing | missing |  |  | - / - |
| full_b12_16384 | missing | missing |  |  | - / - |

## checkpoint: #4656 vs #4780, AC off, dim 2048, blocks of 12

| config | l4656 peak | c4780 peak | saved (c4780 vs l4656) | step 3 loss / gn equal | tps l4656 / c4780 |
|---|---:|---:|---:|---|---:|
| l24_4096 | missing | missing |  |  | - / - |
| l24_8192 | missing | missing |  |  | - / - |
| l24_16384 | missing | missing |  |  | - / - |
| l48_4096 | missing | missing |  |  | - / - |
| l48_8192 | missing | missing |  |  | - / - |
| l48_16384 | missing | missing |  |  | - / - |
