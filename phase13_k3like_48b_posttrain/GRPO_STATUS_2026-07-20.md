# GRPO status (2026-07-20)

## What works: standalone GRPO on the titan K3 model

`phase11_rlhf_grpo_infra/grpo_titan_standalone.py` -- the GRPO RL
algorithm running on the Kimi-Linear/AttnRes 194m model, titan-native
(no inference server, no transformers, no sglang):
- Rollout: full-recompute greedy/sampling decode (prompt >64 so KDA
  chunk-mode stays valid; O(n^2) but correct -- the titan model has no
  recurrent KDA decode path, that lives in the sglang overlay).
- Group-relative advantages (GRPO signature): adv normalized within the
  sampled group, range ~[-1.9, 1.8].
- Advantage-weighted policy-gradient update + grad clipping.
- 6 steps, loop runs (reward is a toy even-token verifier; real GSM8K
  exact-match plugs into reward_fn -- the fixture is random-init so any
  real reward would be ~0, hence a MECHANISM demo not a learning demo).

## Why NOT veRL-native GRPO (the three real blockers, all measured)

1. veRL v1 RL `_ROLLOUT_REGISTRY` = {vllm, sglang, trtllm} async servers
   only; no `hf`/naive rollout in the RL path.
2. transformers K3 inference: `modeling_kimi.py` needs transformers
   ~4.57 (`OutputRecorder`), veRL has 5.14.1 -- version conflict.
3. titan model has no KDA recurrent-decode path (training chunk-mode
   only); real serving = the sglang AttnRes overlay (fork submodule,
   PLAN 2).

## Path to veRL-native GRPO (needs a QIU023/verl fork)

Register a SYNC in-process titan rollout in veRL's RL registry (adapt
the ServerAdapter interface to call the titan actor's module with the
full-recompute decode above), OR stand up the sglang overlay. Both are
veRL-side changes that belong in a user-owned QIU023/verl fork
(matching the QIU023/torchtitan + QIU023/sglang pattern). The standalone
loop proves the algorithm + the titan rollout primitive already work;
the remaining work is the veRL rollout-adapter plumbing.
