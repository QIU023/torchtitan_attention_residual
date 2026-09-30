
## pp2vp4: 4656_pp2vp4 rc=1, pra_pp2vp4 rc=1; steps equal (loss and grad norm) 1 / 1
| tree | step 1 loss / grad norm |
|---|---:|
| 4656 | 8.16693 / 30.1250 |
| pra | 8.16693 / 30.1250 |

| rank | peak allocated step 10, #4656 | PR A | saved | max over steps 2 on, #4656 | PR A | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | nan | nan | nan | 75.20 | 75.23 | -0.03 |

Traced step 5, each rank at the action where allocated memory peaks (GiB):
| rank | tree | action | allocated | blocks held | store | stage inputs | stage outputs | sends only | recv buffers | bound tight | bound paper | blocks max over step | tight max | paper max |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
