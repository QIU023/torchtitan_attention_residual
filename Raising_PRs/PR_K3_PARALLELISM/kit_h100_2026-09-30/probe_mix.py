# Probe (09-30): the image-carrying micro-batches the trainer's dataloader yields for DEPR_IMG_PER16=k, in order (run in ~/mep/w/dep: python probe_mix.py 7).
import os, sys
k = sys.argv[1]
os.environ.update(DEPR_DATA="/root/dep_data/t2i1024_k4", DEPR_RES="1024", DEPW_DIM="6144", DEPV_TOWER="k3", DEPR_SEQ="2048",
                  DEPR_NMAX="1", DEPR_IMG_PER16=k, DEPR_LAYOUT="pp4vpp4", DEPR_AC="full")
sys.path[:0] = ["/root/kit/overnight/dep_ratio", "/root/kit/kit_h100_2026-09-29/dep", "."]
import dep_ratio_local as d
cfg = d.w_dep_k25()
tok = cfg.tokenizer.build(tokenizer_path=cfg.hf_assets_path)
dl = cfg.dataloader.build(dp_world_size=1, dp_rank=0, tokenizer=tok, max_context_length=cfg.training.max_context_length,
                          num_tokens_per_microbatch=2048)
it = iter(dl)
row = []
for i in range(22 * 16):
    mb = next(it)
    g = mb.model_kwargs.get("grid_thw")
    row.append(0 if g is None else int(g.shape[0]))
c = [1 if x else 0 for x in row]
print(f"k={k} first32 {c[:32]}")
print(f"k={k} per-16 counts {[sum(c[j:j+16]) for j in range(0, len(c), 16)]}")
print(f"k={k} M8 steps 1-20: {sum(c[:160])}/160 = {sum(c[:160])/160:.3f}; M16 steps 1-20: {sum(c[:320])}/320 = {sum(c[:320])/320:.3f}; target {int(k)/16:.3f}")
os._exit(0)
