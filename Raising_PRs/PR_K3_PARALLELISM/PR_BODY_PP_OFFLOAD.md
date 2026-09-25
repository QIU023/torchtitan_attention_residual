# PR 4765 body (draft PR, `k3_pp_offload`), rewritten 2026-09-25

Title (unchanged): `[DO NOT review, stack on K3 text PP PR #4312] [Kimi K3] PP ranks activation memory offload`

## Status (not for pasting)

- **Branch.** On 2026-09-25 the user said to sync the draft PRs directly. `k3_pp_offload` was force-pushed from `c73e17c03` to `pp_offload_review1` `49117a146`; the old head is pinned as `backup/k3_pp_offload_pre_20260925`. GitHub shows 17 commits, 25 files.
- **What the head carries on upstream main `9e159aed7`:**
  - #4312's 12 commits (`dbd13d397`);
  - `901ef34de`, the `layers_per_stage` fix that is still only on `pp_review4`;
  - the three rank-store commits of `pp_review_optimize` (`14cba2237`, `202b6a974`, `81b30fd88`);
  - the offload commit.
- **CPU on this head:** the command in the test plan, 96 passed, no skips (`/workspace/venv_bfx9`, torch 2.15.0.dev20260906).
- **GPU smoke, 8 x RTX 5060 Ti, identity only.** Memory numbers stay out of the body until H100 runs.
  - Rank store: `pp_review4` against `202b6a974` over 100 steps at seq 3584, loss and grad norm bitwise (`PP_OPTIMIZE_REPORT_2026-09-24.md` §2.4).
  - Offload: the whole stack at `49117a146`, off and on, over 3 steps, seq 2048 with full AC and seq 512 without AC; loss and grad norm bitwise within each group (`PP_ACTIVATION_STORAGE_REPORT_2026-09-25.md` §3).
  - The third rank-store commit (`81b30fd88`) has no GPU run against `pp_review4`. Its bitwise coverage is the CPU `test_kimi_k3_pp_block_grads`.
- **Diff audit 09-25 (automated greps only)** on `901ef34de..46692171b`: no logbook paths, no measured values in code, no "we"; docstrings are one line, and config fields follow titan's field-docstring style.

--- PR 4765 body: PASTE BEGIN ---

## Summary

Kimi K3 pipeline ranks can park the saved activations they hold longest in pinned host memory and read them back one layer ahead in backward, following the unified activation manager of the Kimi K3 report, on a rank store that receives in place and hands out views.

- `ActivationStorage`, `StorageBackend`, `HostBackend` (`torchtitan/distributed/activation_storage.py`, new): route each tensor autograd saves to a backend chosen per tensor through `torch_remat.saved_tensors_hooks`, and read the saves back one layer ahead in backward.
- `PPOffloadKnobs`, `ActivationPlan` (`kimi_k3/pipeline_parallel/activations.py`, new): pick the stage micro-batches a rank holds longest from the schedule's `pipeline_order`, and the action at which each starts coming back.
- `KimiK3Model.Config.pp_offload` (`kimi_k3/model.py`), wired in `kimi_k3/pipeline_parallel/__init__.py` and `stage.py`; off by default.
- The rank store (`kimi_k3/pipeline_parallel/stage.py`, `cache.py`; the three commits after #4312): the delta is received into the store, stacks and payloads are views of it, receive buffers are allocated on demand, and each send is waited at the first action that proves its receiver consumed it.

## Design

Every tensor saved inside a stage's forward goes through the pack hook of `torch_remat.saved_tensors_hooks`, which titan's activation checkpointing already runs on. The capture context records (stage, micro-batch, layer), so the policy decides per tensor and the model code does not change. A moved tensor is copied out on a storage stream and stays referenced until the copy finishes, so every device allocation stays in the compute stream's pool. In backward, the first unpack of a layer releases the layer after it and starts reading back the layer before it; the plan starts the first layer of each backward `lead` actions early.

Only tensors that own their whole storage move, since moving a view copies without freeing its base. The AttnRes blocks and each stage's inputs stay on the device, as in the report. The plan ranks stage micro-batches by saves times the actions between their forward and backward, and skips those whose backward follows too closely to copy out and back. It needs the action-list schedules (Interleaved1F1B and the like) and refuses the others.

torch's pipelining runtime waits send works only after the step's last action, and a pending work keeps its tensor alive (for a view, the whole storage). Its receive buffers are allocated per micro-batch and kept. So the stage receives the delta straight into the rank store, allocates a receive buffer when the receive is posted, waits its forward sends at its own backward, and waits its input-gradient sends at the first forward whose input the receiver produces after consuming them.

The storage and the host backend live in `torchtitan/distributed` because nothing in them is specific to Kimi K3. The plan lives in the K3 pipeline package because it reads the stage-to-layer layout.

## Relation to #4312 and #4764

Stacked on #4312. The three rank-store commits are not part of #4312. #4764 adds a remote backend to the same storage, to balance activations across PP ranks.

## Test plan

- `pytest tests/unit_tests/cpu/test_activation_storage.py tests/unit_tests/cpu/test_kimi_k3_pp_offload.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py tests/unit_tests/cpu/test_config_manager.py tests/unit_tests/cpu/test_no_new_cli_options.py -q` (96 passed)
  - `test_activation_storage.py`: gradients bitwise with and without the storage, under titan's full AC and under `remat.checkpoint`; nothing read back late after the first-layer prefetch.
  - `test_kimi_k3_pp_offload.py`: two ranks on gloo with the real Interleaved1F1B schedule; loss and every gradient bitwise with 2 and with all stage micro-batches moved.
  - `test_kimi_k3_pp_block_grads.py`: four ranks on gloo with Interleaved1F1B, cache on and off; every block gradient bitwise with one device, and the store empty after an eval pass.
- 8 x RTX 5060 Ti smoke, pp8 x vp2 on local debug flavors: the first two rank-store commits leave loss and grad norm bitwise with #4312 over 100 steps; on the full stack, `pp_offload` on leaves them bitwise with `pp_offload` off over 3 steps, with and without full AC. Peak memory per PP rank on H100 goes here once measured.

--- PASTE END ---
