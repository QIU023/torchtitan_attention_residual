# Text GRPO, three steps: what it proves and what it cannot

`grpo_k3_multistep.sh`, 3 steps, exit 0.

| step | actor/entropy | rollout_probs_diff max | mean |
|---|---|---|---|
| 1 | 7.505845 | 1.503e-03 | 3.030e-04 |
| 2 | 7.506036 | 1.568e-03 | 2.832e-04 |
| 3 | 7.506004 | 1.780e-03 | 2.992e-04 |

`rollout_probs_diff_valid: 1` at every step. `rollout_corr/kl` 0.249, `k3_kl` 0.191.
Memory stable: 2.16 GB allocated on device, 34.5 GB host.

## Established

The loop runs for more than one step without crashing, and the cross-engine logprob
agreement between the rollout engine (vLLM) and the training engine (torchtitan) is
~1.5e-03 and does not drift over three steps. That is worth having: it says the two
engines agree about the same tokens under the same weights, which is the thing a
mismatched tokenizer, layout or dtype would break loudly.

## NOT established, and the reason is exact

**Weight sync is still untested.** From the same step's metrics:

    actor/pg_loss: 0.0    actor/loss: 0.0    actor/grad_norm: 0.0
    critic/rewards/mean: 0.0

`grad_norm` is exactly zero, so the actor never changed. The chain is deterministic:
gsm8k scores by exact match, the fixture is randomly initialised so it never matches,
every reward is 0, GRPO normalises advantages within a group, all-equal rewards give
all-zero advantages, and a zero advantage gives zero policy gradient.

With the weights unchanged, a working sync and a no-op sync produce **identical**
`rollout_probs_diff`. So that metric cannot distinguish them, and reading it as
evidence for the sync -- which I did on first sight -- is wrong.

This is the same failure shape as the earlier probe that recorded only
disagreements: a measurement whose reading is the same whether or not the mechanism
under test did anything. The tell here was available in the same log line, in
`grad_norm: 0.0`.

## What makes it testable

A reward with VARIANCE within each GRPO group. Not merely non-zero -- a constant
non-zero reward still normalises to zero advantage. Once advantages vary, the actor
moves, and then `rollout_probs_diff` after a sync measures something: it compares
the rollout engine's probabilities against a training engine whose weights have
actually changed since the rollout.

The clean check on top of that is a differential one: run a step with the sync
disabled and confirm `rollout_probs_diff` GROWS. If it does not grow when the sync is
removed, the metric is not sensitive to the sync and no value of it means anything.

## Also worth recording

`trainer.total_training_steps=1` had hidden all of this, because step 1 has no
preceding sync at all. The open item in the handoff said "multi-step stability, a task
with non-zero reward, and weight sync measured in-loop" as three items; they are not
independent -- the third is unreachable without the second.
