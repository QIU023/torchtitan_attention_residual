
## dp2pp2vp2: 4656_dp2pp2vp2 rc=1, pra_dp2pp2vp2 rc=1; steps equal (loss and grad norm) 1 / 1
| tree | step 1 loss / grad norm |
|---|---:|
| 4656 | 8.14469 / 22.1250 |
| pra | 8.14469 / 22.1250 |

| rank | peak allocated step 10, #4656 | PR A | saved | max over steps 2 on, #4656 | PR A | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | nan | nan | nan | 72.44 | 72.37 | 0.07 |

Traced step 5, each rank at the action where allocated memory peaks (GiB):
| rank | tree | action | allocated | blocks held | store | stage inputs | stage outputs | sends only | recv buffers | bound tight | bound paper | blocks max over step | tight max | paper max |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
