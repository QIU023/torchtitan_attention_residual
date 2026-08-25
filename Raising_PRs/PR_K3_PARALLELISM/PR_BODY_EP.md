Enables expert parallelism for Kimi K3 on the existing all-to-all token
dispatcher. Three pieces: set_moe_sharding_config declares the routed-expert
layout (w1_EFD S(1), w2_EDF S(2), w3_EFD S(1)) with set_decoder_sharding_config
above it so the activations reaching the MoE boundary are DTensors that can be
redistributed onto the expert mesh; parallelize builds the efsdp mesh, the
expert-data-parallel mesh that excludes the expert axis, and hands it to
apply_fsdp_to_decoder with ep_degree -- the shape deepseek_v3 already resolves;
and expert parallel comes off the unsupported list.

The one non-obvious line is enable_sp. With EP on, the tensor-parallel axis
becomes a token axis inside the MoE region because the sparse mesh folds it into
efsdp, so asking for the sequence-parallel layouts whenever SP is on requests
S(1) -> P(sum), which DTensor rejects. It is therefore derived as enable_ep and
enable_tp rather than passed a constant. Sequence parallel is not offered
separately: this model's CP is not ShardingConfig-driven, so SP and CP would
describe the same axis from two places.

EP shards experts inside the data axis, so each cell is compared against the
pure-data-parallel run at the same world size. kimi_k3_debugmodel_text, seed 42,
--debug.deterministic, one seed checkpoint loaded by every cell:

<<TABLE_EPDP>>

MoonEP is not here. The report's balanced EP with online redundant-expert
planning and migration needs its own dispatcher and planning backend; this is
the plain all-to-all path. comm_backend is pinned to "standard" rather than left
a config choice, so no run can silently believe it is on MoonEP.

Without expert_parallel_degree > 1 none of this executes.

Tested: a CPU test on the declaration, including the enable_sp derivation; an
ep2 integration cell.
