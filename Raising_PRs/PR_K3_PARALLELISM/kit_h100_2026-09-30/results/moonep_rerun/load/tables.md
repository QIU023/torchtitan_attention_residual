| cell | rc | loss 1 / 10 / 20 | static placement max / mean (mean, worst) | MoonEP dispatches with S x K rows | mean slots used |
|---|---|---|---|---:|---:|
| num_moonep_nat | rc=0 | 8.23281 / 3.70432 / 3.08361 | 1.52, 2.23 | 0 / 1280 | 1.0 |
| num_moonep_skew | rc=0 | 8.23464 / 3.67228 / 3.08170 | 3.19, 3.92 | 0 / 1280 | 1.9 |
| num_std_nat | rc=0 | 8.23328 / 3.70123 / 3.08158 | 1.51, 2.23 | - | - |
| num_std_skew | rc=0 | 8.23484 / 3.66762 / 3.07788 | 3.19, 3.93 | - | - |

| timing cell | rc | s / step, steps 11 to 30 |
|---|---|---:|
| time_moonep_nat | rc=0 | 0.3080 |
| time_moonep_skew | rc=0 | 0.3064 |
| time_std_nat | rc=0 | 0.3267 |
| time_std_skew | rc=0 | 0.3250 |
