## the committed cell (unseeded, 10 steps)

rc rc=0; loss 8.12923 -> 6.28207; peak mem 0.56 GiB
  9 x DEP bubble backward: 6 ran at a planned slot, 0 drained at step end, 0 forced by the pending bound, 10 slot(s) found nothing pending
  9 x DEP bubble: 2/2 planned encode(s) ran in a bubble, 4 upfront, 2 left inline, 14 idle slot(s) (0 starved, 12 exhausted)
  8 x DEP vision encode: 6 served from the cache, 2 encoded inline
  1 x DEP bubble backward: 0 ran at a planned slot, 0 drained at step end, 0 forced by the pending bound, 16 slot(s) found nothing pending
  1 x DEP vision encode: 0 served from the cache, 8 encoded inline

## bubble on / off (seed 42, deterministic, one warm cache)

on: rc rc=0, steps 10
off: rc rc=0, steps 10
on2: rc rc=0, steps 10
off2: rc rc=0, steps 10
on vs on2: identical on all 10 steps
off vs off2: identical on all 10 steps
on vs off: first differs at step 3
| step | on | off |
|---:|---|---|
| 1 | 8.08804 / 2.1719 | 8.08804 / 2.1719 |
| 2 | 7.89432 / 2.2188 | 7.89432 / 2.2188 |
| 3 | 7.58724 / 3.4219 | 7.58768 / 3.4219 |
| 4 | 7.31631 / 4.7188 | 7.31730 / 4.7188 |
| 5 | 6.92061 / 6.4375 | 6.92022 / 6.4375 |
| 6 | 6.48984 / 7.3750 | 6.48872 / 7.3750 |
| 7 | 6.25977 / 7.9375 | 6.26252 / 7.9688 |
| 8 | 6.00367 / 7.4062 | 6.00731 / 7.4062 |
| 9 | 5.91208 / 7.6875 | 5.90136 / 7.6250 |
| 10 | 6.09526 / 6.4688 | 6.09622 / 6.4688 |

gradients, bubble on vs off:
rank0_step1.pt tower equal 22/22 text equal 79/79 
rank0_step2.pt tower equal 0/22 text equal 79/79 | tower max rel 2.45e-03 0:vision_encoder.layers.1._checkpoint_wrapped_module.norm2.weight
rank0_step3.pt tower equal 0/22 text equal 0/79 | tower max rel 2.54e-02 0:vision_encoder.pos_embed | text max rel 9.42e-02 1:layers.9._checkpoint_wrapped_module.moe.router.gate.weight
rank1_step1.pt tower equal 0/0 text equal 132/132 
rank1_step2.pt tower equal 0/0 text equal 132/132 
rank1_step3.pt tower equal 0/0 text equal 0/132 | text max rel 3.42e-01 0:layers.0._checkpoint_wrapped_module.ffn_res_proj.weight
rank2_step1.pt tower equal 0/0 text equal 128/128 
rank2_step2.pt tower equal 0/0 text equal 128/128 
rank2_step3.pt tower equal 0/0 text equal 0/128 | text max rel 6.69e-02 1:layers.15._checkpoint_wrapped_module.ffn_res_norm.weight
rank3_step1.pt tower equal 0/0 text equal 104/104 
rank3_step2.pt tower equal 0/0 text equal 104/104 
rank3_step3.pt tower equal 0/0 text equal 0/104 | text max rel 1.01e-01 0:layers.6._checkpoint_wrapped_module.attention_res_norm.weight

## hundred steps, bubble on / off

h100_on: rc rc=0, steps 100
h100_off: rc rc=0, steps 100
first differing step 3
| step | on | off |
|---:|---|---|
| 1 | 8.08804 / 2.1719 | 8.08804 / 2.1719 |
| 2 | 7.89432 / 2.2188 | 7.89432 / 2.2188 |
| 3 | 7.58724 / 3.4219 | 7.58768 / 3.4219 |
| 10 | 5.07263 / 5.0938 | 5.07331 / 5.0625 |
| 20 | 2.89337 / 3.1719 | 2.90563 / 3.2656 |
| 50 | 0.64815 / 2.3125 | 0.67249 / 2.3594 |
| 100 | 0.11921 / 0.7812 | 0.12242 / 0.8086 |
lowest loss on the off run: 0.06305

## mixed data, dp_shard 2 x pp2, even DP ranks text only

mixed: rc rc=0, steps completed 4
mixed_always: rc rc=0, steps completed 4

## step time (no determinism, one warm cache, 30 steps; mean over steps 11-30)

| config | rc | s/step (log timestamps) | tps mean | peak mem (loss rank) | vs dep_off |
|---|---|---:|---:|---:|---:|
| dep_off | rc=0 | 1.0015 | 4092 | 0.48 | +0.0% |
| dep_bubble_off | rc=0 | 0.9174 | 4466 | 0.58 | -8.4% |
| dep_prefetch | rc=0 | 0.9319 | 4397 | 0.58 | -7.0% |
|   | 29 x DEP vision encode: 7 served from the cache, 1 encoded inline | | | | |
| dep_bubble_on | rc=0 | 0.9589 | 4273 | 0.58 | -4.2% |
|   | 29 x DEP bubble backward: 6 ran at a planned slot, 0 drained at step end, 0 forced by the pending bound, 10 slot(s) found nothing pending | | | | |
|   | 29 x DEP bubble: 2/2 planned encode(s) ran in a bubble, 4 upfront, 2 left inline, 14 idle slot(s) (0 starved, 12 exhausted) | | | | |
|   | 28 x DEP vision encode: 6 served from the cache, 2 encoded inline | | | | |
