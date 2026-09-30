
## pp2vp2: 4656_pp2vp2 rc=0, pra_pp2vp2 rc=0; steps equal (loss and grad norm) 100 / 100
| tree | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm |
|---|---:|---:|---:|---:|
| 4656 | 8.06628 / 23.1250 | 5.24328 / 22.1250 | 2.80749 / 4.2188 | 2.45762 / 1.7578 |
| pra | 8.06628 / 23.1250 | 5.24328 / 22.1250 | 2.80749 / 4.2188 | 2.45762 / 1.7578 |

| rank | peak allocated step 10, #4656 | PR A | saved | max over steps 2 on, #4656 | PR A | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 67.85 | 66.60 | 1.25 | 67.85 | 66.60 | 1.25 |
| 1 | 49.77 | 48.75 | 1.02 | 49.77 | 48.75 | 1.02 |

Traced step 5, each rank at the action where allocated memory peaks (GiB):
| rank | tree | action | allocated | blocks held | store | stage inputs | stage outputs | sends only | recv buffers | bound tight | bound paper | blocks max over step | tight max | paper max |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4656 | 2B14 | 67.85 | 1.31 | 0.00 | 0.06 | 0.06 | 1.19 | 0.00 | 0.10 | 0.16 | 1.37 | 0.18 | 0.25 |
| 0 | pra | 2B13 | 66.60 | 0.16 | 0.04 | 0.00 | 0.04 | 0.08 | 0.06 | 0.08 | 0.20 | 0.25 | 0.18 | 0.25 |
| 1 | 4656 | 3B15 | 49.77 | 0.98 | 0.00 | 0.04 | 0.08 | 0.86 | 0.06 | 0.12 | 0.20 | 1.13 | 0.18 | 0.25 |
| 1 | pra | 3B15 | 48.75 | 0.16 | 0.00 | 0.04 | 0.08 | 0.04 | 0.06 | 0.12 | 0.20 | 0.21 | 0.18 | 0.25 |
