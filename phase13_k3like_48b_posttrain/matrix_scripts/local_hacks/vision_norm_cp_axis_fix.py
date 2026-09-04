import pathlib
p=pathlib.Path('torchtitan/models/kimi_k3/sharding.py'); s=p.read_text()
n=s.count('invariant_norm_config()'); assert n==2, n
s=s.replace('invariant_norm_config()','invariant_norm_config(include_cp_axis=True)')
p.write_text(s); print("norms carry the cp axis:", n)
