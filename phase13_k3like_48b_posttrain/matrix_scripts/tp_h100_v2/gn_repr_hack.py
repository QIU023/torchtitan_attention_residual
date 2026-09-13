"""LOCAL PROBE HACK (not committed): GN_REPR=1 prints the step's total grad norm at full precision on rank 0
("GNREPR <step> <value>"), since the logged value has four decimals. argv[1] = tree."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "torchtitan/trainer.py"; s = p.read_text()
if "GN_REPR" in s:
    print("already patched"); sys.exit(0)
anchor = "            # Only the last PP stage owns the loss. First combine its DP/CP\n"
assert s.count(anchor) == 1, s.count(anchor)
hack = ('            if os.environ.get("GN_REPR") == "1" and torch.distributed.get_rank() == 0:  # LOCAL PROBE HACK (not committed)\n'
        '                print(f"GNREPR {self.step} {float(grad_norm.full_tensor() if hasattr(grad_norm, \\"full_tensor\\") else grad_norm):.10e}", flush=True)\n')
p.write_text(s.replace(anchor, hack + anchor, 1)); print("patched", p)
