"""LOCAL PROBE HACK (not committed): PRECLIP_SYNC=1 runs torch.cuda.synchronize() right before clip_grad_norm_,
so the norm cannot read a gradient whose reduce-scatter is still in flight on another stream. argv[1] = tree."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "torchtitan/trainer.py"; s = p.read_text()
if "PRECLIP_SYNC" in s:
    print("already patched"); sys.exit(0)
anchor = "            grad_norm = dist_utils.clip_grad_norm_(\n"
assert s.count(anchor) == 1, s.count(anchor)
hack = '            if os.environ.get("PRECLIP_SYNC") == "1":  # LOCAL PROBE HACK (not committed)\n                torch.cuda.synchronize()\n'
p.write_text(s.replace(anchor, hack + anchor, 1)); print("patched", p)
