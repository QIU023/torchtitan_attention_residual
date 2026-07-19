# Pre-RFC GPU verification checklist (8x RTX 5090)

> Purpose: re-verify the RFC's "Finished work" claims on GPU before posting.
> Scope decision (2026-07-19): only **unit tests + train smoke + PP pressure
> matrix** need re-verification on the current fork (`a3b3c74b3`). The 12.5K
> pretrain runs and the LLaVA three-stage pipeline are **historical evidence**
> — the RFC links point at their reports; do not re-run.

## Already verified locally (Windows CPU — done 2026-07-18)

- Fork suite: **43 passed / 2 skipped** (kimi layers/model/multimodal/model-spec/
  pipeline wiring + AttnRes primitive + all-flavor registry sweep, 113 configs).
- Dense carrier vs current fork: **64 passed / 6 skipped** (incl. the
  1460-line PP adapter grid on fake-PG).
- compileall clean. fla 0.5.1 CPU fallback exercised.
- The 8 remaining skips are hardware gates: fla **triton** real path,
  FlexAttention backward (CUDA), fp8 flavor (SM89+). These are exactly what
  the GPU pass must close.

## Hardware

8x RTX 5090 is sufficient — it is the original pressure-test hardware; every
config below was sized for 32 GB cards. H200 is NOT needed for this checklist
(first hard H200 requirement is the 48B full-param POC leg; see PLAN §3c).

## Steps (~1 day total)

1. **Env + full suites** (~15 min)
   ```bash
   pip install fla-core pytest   # torch 2.11+cu assumed on the box
   cd $WS/torchtitan && pytest torchtitan/experiments/kimi_k3/tests/ -v
   cd $WS && PYTHONPATH="$WS/torchtitan:$WS" pytest phase3_attnres_pp_integration/dense_carrier/ -v
   ```
   Expect: all local skips turn green (fla triton path, flex backward, fp8).

2. **5-step train smokes** (~10 min) — validates the merged Trainer chain +
   the launcher migration (PLAN §4's pending gate):
   ```bash
   # kimi (fla triton real path), smallest flavor, 1 GPU
   torchrun --nproc_per_node=1 -m torchtitan.train \
       --module kimi_k3 --config kimi_linear_194m_block_attn_res \
       --training.steps 5
   # dense via the relocated logbook package
   PYTHONPATH="$WS:$PYTHONPATH" torchrun --nproc_per_node=1 -m torchtitan.train \
       --module phase3_attnres_pp_integration.dense_carrier \
       --config llama3_175m_baseline --training.steps 5
   ```

3. **PP smoke** (~30 min): `launch_4gpu_naive.sh` + `launch_4gpu_adapter.sh`,
   100 steps each (launchers already point at the dense_carrier module).

4. **Pressure matrix re-run** (~half day) — re-verifies RFC numerics bullets:
   - `run_pp_pressure_test.sh` (L16, 3 shapes x naive/adapter, 1000 steps).
   - `run_kimi48b_d1280_e16_L32N8_pp8vp4_adapter.sh` + its naive counterpart,
     300 steps each.
   Acceptance: naive-vs-adapter |dLoss| within the bf16 seed-vs-seed
   nondeterminism band (reference: report max 0.011).

5. **Optional hardening** (~40 min): 447m TP=2 / EP=2 smokes (50 steps);
   fp8 flavor (`kimi_linear_447m_aligned_block_attn_res_n4_fp8`) 5-step smoke
   (SM 12.0 native fp8 — also validates the new ModelConfigConverter path).

## Known risk on SM 12.0

fla `fused_norm_gate` device-side assert history (~every 650 steps under KDA;
PR13 / `hot_patches/` in this repo). All smokes above sit below that
threshold; the 1000-step L16 runs are dense (no KDA); the kimi48b runs are
300 steps. If it fires anyway: switch AC to `full` mode (per the note in
`parallelize.py`) or apply the hot patch.

## Exit

- All green, no code change -> post the RFC as-is (links pinned `a3b3c74b3`).
- Any fix required -> commit, re-pin the RFC's fork links to the new SHA, post.
