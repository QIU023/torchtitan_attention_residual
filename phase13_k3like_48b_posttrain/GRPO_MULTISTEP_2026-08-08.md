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

---

## With a variance reward: the actor moves, and the sync becomes measurable

`grpo_variance_reward.surface_form_reward` wired through veRL's existing
`custom_reward_function` extension point -- no core change. It is a deterministic
hash of the response text, so two samples in a GRPO group differ, the group's
advantages are not all equal, and the value is reproducible enough for a differential
A/B. It is explicitly not a quality signal.

| step | rewards/mean | **grad_norm** | pg_loss | probs_diff max |
|---|---|---|---|---|
| 1 | 0.5117 | **23.637** | 0.0 | 1.749e-03 |
| 2 | 0.5142 | **24.385** | 0.0 | 2.087e-03 |
| 3 | 0.4902 | **24.054** | 0.0 | 2.454e-03 |

`grad_norm` left zero, so the causal chain identified above was right: no reward
variance was the reason the actor never moved, not anything in the engine.

And `rollout_probs_diff_max` now **grows monotonically** across steps, 1.75e-03 to
2.45e-03, which is what changing weights between rollout and training looks like --
the rollout ran on the pre-update policy.

## A signal that would have misled, and did not only because grad_norm was next to it

`pg_loss` is **0.0 while grad_norm is 23.6**, and that is expected rather than
contradictory. GRPO normalises advantages to zero mean within a group, and at the
first inner step the importance ratio is 1, so `pg_loss = -mean(A) = 0` BY
CONSTRUCTION while its gradient is not zero at all.

So for GRPO, `pg_loss` is not a health signal and `grad_norm` is. Reading pg_loss
alone here would have produced "still not learning" -- the same wrong conclusion as
before, from a different metric.

## Still not a proof of the sync

What holds: the actor updates, and with the sync enabled `rollout_probs_diff` stays
around 2e-03 rather than diverging.

What is missing is the differential: disable the sync and `probs_diff` must grow
substantially. If it does not, the metric is insensitive to the sync and no value of
it means anything -- which was true by construction in the zero-reward run and has to
be re-established now that the quantity is non-trivial.
