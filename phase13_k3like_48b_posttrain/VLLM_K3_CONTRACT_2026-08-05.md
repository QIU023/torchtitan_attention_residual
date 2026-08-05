# The weight-sync contract, verified against vLLM's official K3 implementation

veRL's RL path has no naive rollout -- `_ROLLOUT_REGISTRY` holds only
vllm/sglang/trtllm async servers -- so K3 GRPO needs an engine that both has the
model and accepts the names our trainer exports. This checks the second half
against the real implementation.

## Which vLLM is the official one

`vllm/models/kimi_k3/` -- a self-contained package with `nvidia/`, `amd/` and
`common/` variants, registering `KimiK3ForConditionalGeneration` and
`KimiK3MTPModel`, and re-pointing `KimiLinearForCausalLM` at itself so a
text-only K3-family checkpoint loads through the same code.

**The released wheel does not have it.** vLLM 0.26.0, the latest release and what
`pip install vllm` gives, ships only `kimi_linear.py`, `kimi_k25.py`,
`kimi_vl.py`, `kimi_audio.py`. Pointing a rollout at that silently tests the
PREDECESSOR: `KimiLinearForCausalLM` exists there too, as the published
Kimi-Linear-48B class, and it is a different model.

Upstream state, checked today: `vllm/models/kimi_k3` **is on main** (main at
`0187f4c88e`, 2026-08-05). Our checkout is the `kimi-k3` branch at `6dc76a9ade`
(2026-07-28), now a week behind. When `test_vllm_weight_contract.py` was written
the branch was still open as PR #50000, and its docstring says so; that is now
stale in our favour.

## Environment: one venv, no disruption to the matrix env

veRL's vllm rollout imports vllm in-process (`vllm_async_server`, plus a version
check at package import), so verl and the K3-capable vllm must share a venv.

| venv | torch | vllm | K3 |
|---|---|---|---|
| `/venv/main` | 2.14.0.dev20260802+cu130 | 0.26.0 (pip) | no |
| `/venv/vllm_k3` | 2.13.0+cu130 | source `0.1.dev1+g6dc76a9ad` | yes |

verl went into `/venv/vllm_k3` editable with `--no-deps`, plus the pure-Python
deps it needs. **torch stayed at 2.13.0 throughout** and `/venv/main` was not
touched, so nothing about the published matrices moved. torchtitan imports
cleanly on torch 2.13 (model, trainer, parallelize) once tyro, spmd_types,
torchdata, datasets and tensorboard are present.

One veRL fix was needed, in our fork: `verl/third_party/vllm/__init__.py`
rejects a source build outright, because setuptools-scm reports `0.1.dev1+g<sha>`
for a checkout with no release tag and `vs.parse(...) >= 0.7.0` is false.
`VERL_VLLM_VERSION` now overrides it. Generic to anyone building vLLM from
source, so it is upstreamable as-is.

## Result: the contract holds

    engine built, hf_config=KimiLinearConfig

All **1032 exported tensors accepted** by the official loader. That covers the
parts most likely to be wrong, and each was confirmed by getting past it rather
than by reading:

* Block AttnRes, two per layer -- `self_attention_res_proj/norm` plus
  `mlp_res_proj/norm`. The official inference implementation **settles the
  per-layer-vs-per-sublayer question we had listed as unsettled**: it is two per
  layer, which is what we already had.
* The final aggregation over block representations, `model.output_attn_res_*`,
  independently confirming the retraction we made about the eager PR.
* The fused KDA input projection: our unfused `q_proj` / `k_proj` / `v_proj` /
  `b_proj` / `f_a_proj` / `g_proj` are consumed by its `stacked_params_mapping`
  into `in_proj_qkvgfab`, and `q_conv1d`/`k_conv1d`/`v_conv1d` into `conv1d`.
* Stable LatentMoE: `routed_expert_down_proj` / `routed_expert_up_proj` /
  `routed_expert_norm`.
* Per-expert `w1`/`w2`/`w3` sliced out of our stacked `w1_EFD`/`w2_EDF`/`w3_EFD`.

## Three defects found getting there, all ours

**1. The export wrote our own names.** `export_vl_to_hf.py` dumps DCP keys
verbatim, so a multimodal export carried `attn_res_proj` and
`final_attn_res_proj` where the release has `self_attention_res_proj` and
`output_attn_res_proj`. `hf_key_map.titan_to_official` has had the right mapping
all along; the export simply never called it. `export_text_to_hf.py` now routes
every key through it and refuses to write if any key has no official name, rather
than dropping it.

**2. The config had two sources of truth for the KDA gate, and they disagreed.**
The stale fixture carried `kda_use_full_rank_gate: false` at top level (what our
model reads) and `linear_attn_config.use_full_rank_gate: true` (what the official
loader reads), with low-rank `g_a_proj`/`g_b_proj` weights. The loader computed
full-rank, looked for `g_proj`, and raised `KeyError:
layers.0.self_attn.g_a_proj.weight` -- a config bug wearing a naming bug's
clothes. Fixed at the source: `titan_config_to_official` derives config and
names from one `KimiK3Config`.

The fixture was also stale in a second way: `hidden_act: silu` where K3 is SiTU-GLU.

**3. The text-only prefix.** `hf_key_map` targets the released layout, which is
multimodal, so it emits `language_model.model.*`. `KimiLinearForCausalLM`'s
parameters are `model.*` -- vLLM's multimodal class is what adds
`language_model.`. A text-only export with the wrapper prefix fails with "no
module or parameter named 'language_model'".

## Resolved: the source build, and the smoke passes unshimmed

    engine built, hf_config=KimiLinearConfig
    prompt_tokens=20 output_ids=[740, 3268, 3419, 64, 1246, 2459, 1005, 2781]
    ROLLOUT SMOKE: PASS

No shim, real kernels, generation end to end through the official K3
implementation on weights our exporter wrote. The output is noise because the
fixture is random-init; the pass condition is that the official class accepted
every weight and the KDA/MLA decode path ran.

Build recipe that worked (`/workspace/build_vllm_k3.sh`):

    unset VLLM_USE_PRECOMPILED
    export TORCH_CUDA_ARCH_LIST="12.0"   # sm_120, the only GPU here
    export MAX_JOBS=32 NVCC_THREADS=4
    export PATH="$HOME/.cargo/bin:$PATH"
    uv pip install --python /venv/vllm_k3/bin/python --no-build-isolation -e .

Two prerequisites that were absent and are easy to lose an hour to: the build
needs `setuptools_rust` **and a Rust toolchain** (neither `cargo` nor `rustc`
existed on this box; rustup, minimal profile), plus `setuptools_scm`. 404 CUDA
objects, ~65 min wall clock at `MAX_JOBS=32`, ~3G of extra disk.

Checked before starting rather than after: FlashKDA's cmake gates on
`9.0a` / `10.0f` / `12.0f`, so sm_120 is supported -- confirmed in the log by
`FlashKDA CUDA architectures: 12.0f`. On a GPU outside that set the whole build
would have been wasted.

`situ_and_mul` lives in the stable-libtorch extension, so it registers on
`import vllm._custom_ops`, not on `import vllm._C` (which does not exist in this
build). `situ_op_shim.py` is kept only as a record of the precompiled-install
failure mode; nothing needs it now.

## The blocker before that build, for the record

Generation fails after a successful load:

    ModuleNotFoundError: No module named 'vllm._flashkda_C'

and, at construction, `_C.situ_and_mul`. Both are compiled extensions **added by
the K3 branch**, and `/workspace/vllm_k3` was installed with
`VLLM_USE_PRECOMPILED=1`, which reuses a release wheel's binaries. The release
has no K3, so it has neither kernel.

`situ_and_mul` is worth a note upstream: `SituAndMul.__init__` does
`self.op = torch.ops._C.situ_and_mul` unconditionally under
`current_platform.is_cuda_alike()`, so construction fails before CustomOp
dispatch can choose `forward_native` -- which exists and computes the same
thing. A lazy lookup would let a precompiled install fall back instead of dying.
Our `situ_op_shim.py` registers a Python implementation to get past it; that is a
contract check, explicitly not a fidelity or performance run.

FlashKDA has no such fallback, and our own notes already record that it is the
official inference kernel and requires safe_gate. So the next step is a real
source build of vLLM with kernels compiled -- a long compile, and disk is at 93%
(28G free), so it wants a deliberate decision rather than being slipped in.

Nothing about GRPO itself is blocked on design any more: the trainer side already
exports per-tensor params through `get_per_tensor_param` (HF conversion, EP
all-gather, `lm_head.weight` re-added under weight tying), and the rollout side
now provably accepts what that produces.

## Reproduce

    # export a fixture through both halves of the contract
    PYTHONPATH=<titan> python export_text_to_hf.py \
        --flavor kimi_k3_k3mini_block_attn_res --out /workspace/k3mini_text_hf \
        --vocab-size 4096

    # load it in the official implementation
    CUDA_VISIBLE_DEVICES=0 K3_SITU_SHIM=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      VERL_VLLM_VERSION=0.27.0 PYTHONPATH=. \
      /venv/vllm_k3/bin/python vllm_k3_rollout_smoke.py /workspace/k3mini_text_hf
