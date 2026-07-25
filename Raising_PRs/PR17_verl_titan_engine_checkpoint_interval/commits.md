# Source commits (fork verl `kimi_k3_integration`)

| commit | scope |
|---|---|
| `c2c2c5a8` | `[kimi_k3] torchtitan engine: honor verl save_freq by setting checkpoint interval=1` -- the whole PR; 1 file, +5 lines. Nothing kimi-specific despite the prefix. |

Patch: `checkpoint_interval_PR17.patch` (applies to `verl/workers/engine/torchtitan/transformer_impl.py`).

Upstream state verified 2026-07-25 against `volcengine/verl` main @ `983cb0f2`: `CheckpointManager.Config` in `TorchTitanEngine.__init__` still has no `interval` argument.
