| cell | rc | loss 1 / 10 / 20 | static placement max / mean (mean, worst) | MoonEP dispatches with S x K rows | mean slots used |
|---|---|---|---|---:|---:|
   num_moonep_nat: nonzero rows per dispatch min 4096 mean 4096.0 max 4096 (S x K = 4096), 1280 / 1280 exactly S x K; padded minus the zero-fill counts: min 4096 max 4096
| num_moonep_nat | rc=0 | 8.23281 / 3.70515 / 3.08309 | 1.54, 2.26 | 1280 / 1280 | 1.0 |
   num_moonep_skew: nonzero rows per dispatch min 4096 mean 4096.0 max 4096 (S x K = 4096), 1280 / 1280 exactly S x K; padded minus the zero-fill counts: min 4096 max 4096
| num_moonep_skew | rc=0 | 8.23464 / 3.66700 / 3.08168 | 3.19, 3.92 | 1280 / 1280 | 1.9 |
| num_std_nat | rc=0 | 8.23328 / 3.69303 / 3.08730 | 1.51, 2.22 | - | - |
| num_std_skew | rc=0 | 8.23484 / 3.66723 / 3.08044 | 3.19, 3.93 | - | - |

| timing cell | rc | s / step, steps 11 to 30 |
|---|---|---:|
| time_moonep_nat | rc=0 | 0.5190 |
| time_moonep_skew | rc=0 | 0.5335 |
| time_std_nat | rc=0 | 0.6384 |
| time_std_skew | rc=0 | 0.6483 |
