# PP Adapter Pressure Test — Plan (2026-05-11)

## TL;DR

Single-node PP=8 + VP=4 stress test on the existing `CrossStageCacheAdapter`
infrastructure, using our 447M aligned Kimi AttnRes model as carrier (more
realistic than the 175m Llama3 attn_res toy used in earlier phase3 runs).
**Cannot start until Stage 3 GRPO finishes** (~13:45 today). Doc captures
test matrix, config, hardware budget, expected on-wire signatures.

## What's already done in phase3

| Run | PP | Adapter | Status |
|---|---|---|---|
| `pp4_naive_4gpu` | 4 | off | baseline ✓ |
| `pp4_adapter_4gpu` | 4 | on | numerics match naive ✓ |
| `pp8_naive` | 8 | off | baseline ✓ |
| `pp8_adapter` | 8 | on | numerics match naive ✓ |

Carrier: `llama3_175m_attn_res_L16_n8` (16 layers, 8 attn-res blocks of
2 layers each). Adapter implementation: `phase3/adapter.py` + thin wrapper
in `torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py` →
`pipeline_llm_with_cache_adapter`. Zero core-torchtitan changes.

What's NOT yet exercised:
- **VP > 1** (Interleaved 1F1B schedule). Current PP=8 runs are 1F1B
  non-interleaved (VP=1). Real production configs use VP=2-4.
- **Larger model** as carrier. 175m is below the per-stage compute
  threshold where adapter overhead is even visible.
- **Multi-node PP** (PP=16 across two machines, intra/inter-node bandwidth
  asymmetry exercises the cache-staleness logic differently).

## Hardware available

8× RTX 5090 32GB SM 12.0 (Blackwell consumer). Today during Stage 2 SFT
(in flight), each GPU is using ~14.6GB → **~17.4GB free / GPU = 139GB
total** if we wanted to share GPUs with training, but realistically the
PP test wants an exclusive scheduler reservation.

Multi-node not currently available (single host).

## Test matrix

### P0 — single-node PP=8, VP={1,2,4}, 447M carrier

| run | model | PP | VP | µbs | gbs | seq_len | layers/stage | est step time | adapter overhead test |
|---|---|---|---|---|---|---|---|---|---|
| `pp8_vp1_447m_naive` | 447m AttnRes | 8 | 1 | 8 | 64 | 512 | 2 | TBD | baseline |
| `pp8_vp1_447m_adapter` | 447m AttnRes | 8 | 1 | 8 | 64 | 512 | 2 | TBD | overhead = (this - naive)/naive |
| `pp8_vp2_447m_naive` | 447m AttnRes | 8 | 2 | 4 | 64 | 512 | 1 | TBD | baseline (Interleaved 1F1B) |
| `pp8_vp2_447m_adapter` | 447m AttnRes | 8 | 2 | 4 | 64 | 512 | 1 | TBD | overhead at VP=2 |
| `pp8_vp4_447m_naive` | 447m AttnRes | 8 | 4 | 4 | 64 | 512 | 0.5? | TBD | stretch — needs >= 16 layers |
| `pp8_vp4_447m_adapter` | 447m AttnRes | 8 | 4 | 4 | 64 | 512 | 0.5? | TBD | overhead at VP=4 |

**Issue with VP=4 on 16-layer 447M:** PP×VP=32 chunks across 16 layers
means each chunk has half a layer, which doesn't work. Need to either
(a) use a deeper carrier model, or (b) cap at VP=2.

Resolution: run **VP=1, VP=2** on 447M (16 layers), and run **VP=4** on
the 175M attn_res L=16 (so 16 layers / 32 chunks = 0.5 — also fails) or
the kimi_linear 528m which has 24 layers (24/32 = 0.75 — also fails).

**Real fix:** push the carrier up to a model with **≥32 layers**, e.g.
the kimi_linear 1B configuration with 28 layers — still not 32. Or just
stop at PP=8 VP=2 for the 447M (PP×VP=16 = layers, exactly 1 per chunk,
fine).

### P1 — same matrix on the existing 175M L16 attn_res

Same test but the toy 175M carrier from earlier phase3 work. Cheap to
run, validates the test harness before burning compute on 447M. Use this
to debug the VP=2 schedule wiring before committing to the 447M sweep.

### P2 — exit conditions

- **Pass criteria**: adapter step time within **5% of naive baseline** at
  every PP/VP config. Numeric loss curves match within bf16 tolerance
  (1e-3 relative for first 100 steps, same as phase3 baseline criteria).
- **Fail criteria**: > 10% overhead at any config → root cause + fix
  before declaring adapter shippable.
- **Trace requirement**: NCCL trace tier_b (collective + flow summary)
  for each VP config. The interesting signature is **stage-N→stage-(N+1)
  send-bytes** going from O(K_i) (naive) to O(ΔK_i = 1) (adapter) per
  microbatch. Save under `phase3/runs/pp8_vp${VP}_447m_adapter/tier_b_trace/`.

## When can we start

Pipeline state right now (2026-05-11 ~06:25):

```
Stage 1: pretrain step 7500 ✓ DONE
Stage 2: SFT step ~5500/7000, in flight (continuation w/ MAX_RETRIES=30)
Stage 3a: DCP→HF VLM conversion — waiting
Stage 3b: GRPO 1500 steps — waiting
```

Stage 2 ETA ~08:15, Stage 3 ETA ~13:45. **Pressure test can start after
13:45 today.** All 8 GPUs are pinned until then.

Alternative: **interrupt Stage 3 GRPO** (it's framework-validation again,
not a research output by itself). User has previously said no to
placeholder runs — but Stage 3 GRPO on the 5000-step SFT ckpt IS the
real research weights, so probably want to let it finish.

## Launcher to write before start

`phase3/run_pressure_test_447m.sh` — wraps `launch_8gpu_adapter.sh` (already
exists) with:
- carrier flavor = `kimi_linear_447m_aligned_block_attn_res`
- PP / VP grid sweep
- Loops naive then adapter for each PP/VP combo
- Calls `phase7/extract_collectives.py` + `phase7/expand_to_flows.py`
  per-run for NCCL summary

ETA to write: 30 min once Stage 3 starts.

## What we'd report

Single table per Stage 3 finish:

| PP | VP | naive step (s) | adapter step (s) | overhead % | send-bytes ratio | loss-curve match |
|---|---|---|---|---|---|---|
| 8 | 1 | x.xx | y.yy | z.z% | n.nn× smaller | ✓/✗ |
| 8 | 2 | ... | ... | ... | ... | ... |

Plus NCCL trace dirs ready for ixia replay (the phase 7 pattern).
