# GRPO status (2026-07-20)

## UPDATE 2026-07-23 -- GRPO with the sglang overlay as the rollout WORKS (external-server, HTTP)

`phase11_rlhf_grpo_infra/grpo_titan_sglang_rollout.py` runs the full GRPO
loop on the titan K3 194m actor with the user's sglang AttnRes overlay as
the ROLLOUT, end-to-end on the 5090 -- 6 steps, PASS, every step
`weight_sync=True`. Division of labor (the PLAN-2 commitment): the training
parts are titan/ours (group-relative advantages, the titan actor forward +
policy-gradient update, the actor->rollout weight-sync orchestration); the
rollout is the user's sglang server, running in its own venv
(`/workspace/sgl_venv`, torch 2.11) as an EXTERNAL HTTP server -- this
script (torch 2.12 titan venv) talks to it over HTTP only, no sglang import,
so the 2.11-vs-2.12 venv split is bypassed. Rollout = POST `/generate`
(input_ids -> output_ids, verified). Weight sync (veRL has no pure-HTTP
full-weight path) = actor exports HF weights to a shared dir + POST sglang's
native `/update_weights_from_disk` (verified, `{"success":true}`), so the
rollout tracks the actor each step. Random-init 194m => reward ~0.5 toy
signal (mechanism demo, not learning). This is the mechanism the veRL-native
external-server rollout adapter (Path A(b), contract mapped below) will
productize.

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

## veRL fork IS now set up (QIU023/verl, submodule `verl/`)

Synced to upstream volcengine/verl main (6a6242f3), branch
`kimi_k3_integration`, patched (engine mapping + namespace fallback +
gloo backend), SFT verified end-to-end on the fresh mainline. Two paths
remain for veRL-native GRPO rollout, both substantial:

**Path A -- sglang rollout (the DESIGNED path).** veRL already has
`("sglang","async")` in the registry. Install QIU023/sglang
(@attention_residual_inference, the AttnRes inference overlay) and run
`rollout.name=sglang`.

UPDATE 2026-07-20 -- the overlay now PROVABLY serves Kimi-Linear on this
box (RTX 5090 / Blackwell). Installed into an ISOLATED venv
(`/workspace/sgl_venv`, torch 2.11 -- does NOT touch /venv/main or
/venv/verl @ torch 2.12) via `install_sglang_isolated.sh` (needs
rust+protoc for the sglang-grpc crate; torchvision realigned to cu130;
`kernels<0.13` pinned for transformers 5.6). `sglang_serve_smoke.sh`
loads the 194m official-format checkpoint and serves `/generate`
end-to-end (READY in 54s; output is gibberish only because the fixture
is random-init -- the MLA+KDA+MoE serving PIPELINE runs). Two real bugs
fixed en route (sglang fork e45f675), both of which would also break the
real 48B: (1) `moe_fused_gate` fp32/bf16 dtype mismatch on the bf16
`e_score_correction_bias` (verified bf16 on the real 48B weights);
(2) flashinfer CUTE rmsnorm requires an 8-stride-aligned input, MLA
feeds a non-contiguous latent slice.

REMAINING boundary for veRL-native GRPO: veRL's sglang rollout imports
sglang IN-PROCESS (`http_server_engine` top-level import), so veRL and
sglang must share one venv -> torch 2.11 (sglang) vs 2.12 (titan engine)
collide. Two closes, both mapped + de-risked 2026-07-20:

(a) Unify on torch 2.11. De-risked in `/workspace/sgl_venv` (torch 2.11):
    fla-core imports and the KDA chunk kernel RUNS on 5090/Blackwell; the
    titan Kimi-Linear KDA model BUILDS + FORWARDS (added tyro, spmd_types,
    torchdata, datasets, tensorboard, wandb -- none pin torch). The one
    wall: titan's TRAINING stack (`distributed/full_dtensor.py`) uses
    `DataParallelMeshDims` from `torch.distributed.fsdp` -- and PASSES it to
    `fully_shard` -- which is absent in torch 2.11.0 stable (the model path
    is clean; only the FSDP/parallel trainer chain needs it). It is an FSDP
    *API* coupling, not just an import: a fallback NamedTuple fixes the
    import but 2.11.0's `fully_shard` won't accept it. Real close = torch
    2.11 with that symbol (the nightly the fork's "fleet is 2.11" note
    assumes) OR adapting titan FSDP to 2.11.0's API. Not a one-line shim, so
    (b) is the cleaner path on THIS box (torch 2.11.0 stable in sgl_venv).
(b) External-server HTTP rollout: run the standalone sglang server (proven
    above) in its own 2.11 venv and have the 2.12 veRL trainer talk to it
    over HTTP WITHOUT importing sglang in-process. Needs a thin ServerAdapter
    that is pure-HTTP (no `from sglang...` at import), plus the delta_loader
    registered server-side (already present) for weight sync.

The overlay itself is no longer the blocker (it serves); the MODEL runs on
2.11; the wiring is (a) one titan dist-API shim or (b) a pure-HTTP adapter.

**Path B -- in-process titan sync rollout.** A new `BaseRollout` that
holds the actor's live module and does full-recompute decode. Needs
engine_workers to thread the actor module into the rollout (the
ServerAdapter ctor takes only config/model_config/device_mesh) + the
DataProto response-packing format. A veRL worker-architecture change.

## Original notes

Register a SYNC in-process titan rollout in veRL's RL registry (adapt
the ServerAdapter interface to call the titan actor's module with the
full-recompute decode above), OR stand up the sglang overlay. Both are
veRL-side changes that belong in a user-owned QIU023/verl fork
(matching the QIU023/torchtitan + QIU023/sglang pattern). The standalone
loop proves the algorithm + the titan rollout primitive already work;
the remaining work is the veRL rollout-adapter plumbing.
