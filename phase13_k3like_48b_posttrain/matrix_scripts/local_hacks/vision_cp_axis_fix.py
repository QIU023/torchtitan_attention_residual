import pathlib
p=pathlib.Path('torchtitan/models/kimi_k3/sharding.py'); s=p.read_text()
n=s.count('vision_invariant_linear_config()'); assert n>=1
s=s.replace('vision_invariant_linear_config()','vision_invariant_linear_config(include_cp_axis=True)')
old='set_vision_transformer_block_sharding_config(ve_cfg.block, rope_cache_dp=spmd.V)'
assert s.count(old)==1
s=s.replace(old,'''set_vision_transformer_block_sharding_config(
        ve_cfg.block, rope_cache_dp=spmd.V, include_cp_axis=True
    )''')
p.write_text(s)
p=pathlib.Path('torchtitan/models/kimi_k3/model.py'); m=p.read_text()
old='input_sharding = {**decoder_input_sharding(), **multimodal_input_sharding()}'
assert m.count(old)==1
m=m.replace(old,'''input_sharding = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(include_cp_axis=True),
        }''')
p.write_text(m); print("vision declarations carry the cp axis:", n, "linear configs + the block + the inputs")
