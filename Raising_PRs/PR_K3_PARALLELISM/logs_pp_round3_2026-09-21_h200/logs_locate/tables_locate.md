# fp32 locate and exact-norm cells, reference = ref (fp32 lineage, matched accumulation), 100 steps
| cell | step 1 | step 10 | step 15 | step 20 | step 50 | step 100 | step 1 | step 10 | step 15 | step 20 | step 50 | step 100 (grad norm) | steps identical (loss, norm) | first loss difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | `8.055300` | `3.405580` | `3.103010` | `2.916880` | `2.578440` | `2.501960` | `2.155100` | `1.274600` | `1.258900` | `0.936800` | `0.711300` | `0.855800` | - | - |
| ref2 | `8.055300` (bitwise) | `3.405580` (bitwise) | `3.103010` (bitwise) | `2.916880` (bitwise) | `2.578440` (bitwise) | `2.501960` (bitwise) | `2.155100` (bitwise) | `1.274600` (bitwise) | `1.258900` (bitwise) | `0.936800` (bitwise) | `0.711300` (bitwise) | `0.855800` (bitwise) | 100/100, 100/100 | none |
| vp2n | `8.055300` (bitwise) | `3.405580` (bitwise) | `3.103000` (-0.000322%) | `2.916880` (bitwise) | `2.576750` (-0.0655%) | `2.499220` (-0.11%) | `2.155100` (bitwise) | `1.274600` (bitwise) | `1.258800` (-0.00794%) | `0.939200` (+0.256%) | `0.712500` (+0.169%) | `0.860800` (+0.584%) | 15/100, 14/100 | 15 |
| vp2c | `8.055300` (bitwise) | `3.405580` (bitwise) | `3.103000` (-0.000322%) | `2.916880` (bitwise) | `2.576820` (-0.0628%) | `2.499740` (-0.0887%) | `2.155100` (bitwise) | `1.274600` (bitwise) | `1.258800` (-0.00794%) | `0.939200` (+0.256%) | `0.712700` (+0.197%) | `0.861300` (+0.643%) | 15/100, 14/100 | 15 |
| pp4vp4n | `8.055300` (bitwise) | `3.405580` (bitwise) | `3.103000` (-0.000322%) | `2.916880` (bitwise) | `2.576820` (-0.0628%) | `2.499740` (-0.0887%) | `2.155100` (bitwise) | `1.274600` (bitwise) | `1.258800` (-0.00794%) | `0.939200` (+0.256%) | `0.712700` (+0.197%) | `0.861300` (+0.643%) | 15/100, 14/100 | 15 |
| vp2n_xn | `8.055300` (bitwise) | `3.405580` (bitwise) | `3.103010` (bitwise) | `2.916880` (bitwise) | `2.578440` (bitwise) | `2.501960` (bitwise) | `2.155100` (bitwise) | `1.274600` (bitwise) | `1.258900` (bitwise) | `0.936800` (bitwise) | `0.711300` (bitwise) | `0.855800` (bitwise) | 100/100, 100/100 | none |
| vp2c_xn | `8.055300` (bitwise) | `3.405580` (bitwise) | `3.103000` (-0.000322%) | `2.916880` (bitwise) | `2.576820` (-0.0628%) | `2.499740` (-0.0887%) | `2.155100` (bitwise) | `1.274600` (bitwise) | `1.258800` (-0.00794%) | `0.939200` (+0.256%) | `0.712700` (+0.197%) | `0.861300` (+0.643%) | 15/100, 14/100 | 15 |
| pp4vp4n_xn | `8.055300` (bitwise) | `3.405580` (bitwise) | `3.103010` (bitwise) | `2.916880` (bitwise) | `2.578440` (bitwise) | `2.501960` (bitwise) | `2.155100` (bitwise) | `1.274600` (bitwise) | `1.258900` (bitwise) | `0.936800` (bitwise) | `0.711300` (bitwise) | `0.855800` (bitwise) | 100/100, 100/100 | none |

# total grad norm bits per step (GNREPR rank 0, hex) against ref
ref2: 0/100 steps differ; first:
vp2n: 94/100 steps differ; first: 2 4 5 7 8 9 10 13
vp2c: 94/100 steps differ; first: 2 5 7 8 9 10 13 14
pp4vp4n: 93/100 steps differ; first: 2 6 7 8 10 11 13 15
vp2n_xn: 0/100 steps differ; first:
vp2c_xn: 92/100 steps differ; first: 2 5 7 9 10 13 15 16
pp4vp4n_xn: 0/100 steps differ; first:
