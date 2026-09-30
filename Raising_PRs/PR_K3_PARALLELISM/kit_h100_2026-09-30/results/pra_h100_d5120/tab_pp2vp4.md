
## pp2vp4: 4656_pp2vp4 rc=0, pra_pp2vp4 rc=0; steps equal (loss and grad norm) 100 / 100
| tree | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm |
|---|---:|---:|---:|---:|
| 4656 | 8.11475 / 35.2500 | 6.92955 / 23.1250 | 2.77905 / 4.0625 | 2.40688 / 1.1328 |
| pra | 8.11475 / 35.2500 | 6.92955 / 23.1250 | 2.77905 / 4.0625 | 2.40688 / 1.1328 |

| rank | peak allocated step 10, #4656 | PR A | saved | max over steps 2 on, #4656 | PR A | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 67.31 | 65.69 | 1.62 | 67.31 | 65.69 | 1.62 |
| 1 | 52.72 | 50.65 | 2.07 | 52.72 | 50.65 | 2.07 |

Traced step 5, each rank at the action where allocated memory peaks (GiB):
| rank | tree | action | allocated | blocks held | store | stage inputs | stage outputs | sends only | recv buffers | bound tight | bound paper | blocks max over step | tight max | paper max |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4656 | 6B14 | 67.31 | 1.84 | 0.00 | 0.27 | 0.04 | 1.52 | 0.00 | 0.14 | 0.16 | 1.91 | 0.18 | 0.29 |
| 0 | pra | 6B15 | 65.69 | 0.23 | 0.00 | 0.12 | 0.00 | 0.12 | 0.00 | 0.12 | 0.16 | 0.33 | 0.18 | 0.29 |
| 1 | 4656 | 5B13 | 52.72 | 1.99 | 0.12 | 0.16 | 0.12 | 1.60 | 0.06 | 0.20 | 0.27 | 2.50 | 0.21 | 0.33 |
| 1 | pra | 5B13 | 50.65 | 0.31 | 0.08 | 0.08 | 0.04 | 0.12 | 0.06 | 0.20 | 0.27 | 0.33 | 0.21 | 0.33 |
