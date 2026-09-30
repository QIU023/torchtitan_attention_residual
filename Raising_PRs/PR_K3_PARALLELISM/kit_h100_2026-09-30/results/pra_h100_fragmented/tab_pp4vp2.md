
## pp4vp2: 4656_pp4vp2 rc=1, pra_pp4vp2 rc=1; steps equal (loss and grad norm) 4 / 4
| tree | step 1 loss / grad norm |
|---|---:|
| 4656 | 8.08712 / 24.3750 |
| pra | 8.08712 / 24.3750 |

| rank | peak allocated step 10, #4656 | PR A | saved | max over steps 2 on, #4656 | PR A | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | nan | nan | nan | 55.98 | 54.69 | 1.29 |

Traced step 5, each rank at the action where allocated memory peaks (GiB):
| rank | tree | action | allocated | blocks held | store | stage inputs | stage outputs | sends only | recv buffers | bound tight | bound paper | blocks max over step | tight max | paper max |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4656 | 0B8 | 55.98 | 1.69 | 0.02 | 0.21 | 0.23 | 1.22 | 0.07 | 0.30 | 0.45 | 1.78 | 0.35 | 0.52 |
| 0 | pra | 0B8 | 54.69 | 0.54 | 0.02 | 0.21 | 0.07 | 0.23 | 0.07 | 0.30 | 0.45 | 0.59 | 0.35 | 0.52 |
